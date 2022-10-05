import torch
from torch import nn
import torch.nn.functional as F
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss, get_shape_from_act_space
import numpy as np
import copy

# def get_dist_rep_shape(args):
#     '''
#     return an integer, the dims of repre for action distributions
#     '''

#     return 18

    # return args.grid_size ** 2 + 2 * 2

def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])

class DistillationBuffer(SharedReplayBuffer):
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        super(DistillationBuffer, self).__init__(args, num_agents, obs_space, share_obs_space, act_space)
        self.dist_rep_shape = act_space.n
        self.teacher_dist_rep = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self.dist_rep_shape), dtype=np.float32)
        self.teacher_action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        # self.teacher_rnn = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        # self.teacher_rnn_critic = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.num_agents = num_agents
        self.args = args
        self.teacher = None
    
    def set_teachers(self, teacher):
        self.teacher = teacher
    
    def prepare_dists(self):
        torch_grad = torch.is_grad_enabled()
        if torch_grad:
            torch.set_grad_enabled(False)

        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length
        num_mini_batch = self.args.num_mini_batch
        mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        if self.args.teacher_algo == "MAT":
            share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
            share_obs = share_obs[rows, cols]
            obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
            obs = obs[rows, cols]
            rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
            rnn_states = rnn_states[rows, cols]
            rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
            rnn_states_critic = rnn_states_critic[rows, cols]
            actions = self.actions.reshape(-1, *self.actions.shape[2:])
            actions = actions[rows, cols]
            if self.available_actions is not None:
                available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
                available_actions = available_actions[rows, cols]
            value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
            value_preds = value_preds[rows, cols]
            returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
            returns = returns[rows, cols]
            masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
            masks = masks[rows, cols]
            active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
            active_masks = active_masks[rows, cols]

            _, action_log_probs, dist_entropy, dist_rep = self.teacher[0].evaluate_actions(share_obs, 
                                                                                    obs, 
                                                                                    rnn_states, 
                                                                                    rnn_states_critic,
                                                                                    actions,
                                                                                    masks, 
                                                                                    available_actions,
                                                                                    active_masks)
            self.teacher_action_log_probs = action_log_probs.resize(self.episode_length, self.n_rollout_threads, self.num_agents,1).to("cpu").numpy()
            self.teacher_dist_rep = dist_rep.resize(self.episode_length, self.n_rollout_threads, self.num_agents, self.dist_rep_shape).to("cpu").numpy()
        
        else:
            # for e in range(self.n_rollout_threads):
            #     obs = self.obs[:-1, e].reshape(-1, *self.obs.shape[3:])
            #     rnn_states = self.rnn_states[:-1, e].reshape(-1, *self.rnn_states.shape[3:])
            #     if self.available_actions is not None:
            #         available_actions = self.available_actions[:-1, e].reshape(-1, self.available_actions.shape[-1])
            #     masks = self.masks[:-1, e].reshape(-1, 1)
            #     actions = self.actions[:, e].reshape(-1, self.actions.shape[-1])
            #     active_masks = self.active_masks[:-1, e].reshape(-1, 1)

            #     action_log_probs, dist_entropy, dist_rep = self.teacher[e].actor.distill_evaluate_actions(obs,
            #                                                                             rnn_states,
            #                                                                             actions,
            #                                                                             masks,
            #                                                                             available_actions,
            #                                                                             active_masks)
            #     self.teacher_action_log_probs[:, e, :, :] = action_log_probs.resize(self.episode_length, self.num_agents, 1).to("cpu").numpy()
            #     self.teacher_dist_rep[:, e, :, :] = dist_rep.resize(self.episode_length, self.num_agents, self.dist_rep_shape).to("cpu").numpy()
        
            share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
            rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
            rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
            actions = self.actions.reshape(-1, self.actions.shape[-1])
            if self.available_actions is not None:
                available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
            value_preds = self.value_preds[:-1].reshape(-1, 1)
            returns = self.returns[:-1].reshape(-1, 1)
            masks = self.masks[:-1].reshape(-1, 1)
            active_masks = self.active_masks[:-1].reshape(-1, 1)

            rnn_states_teacher = rnn_states.copy()

            action_log_probs, dist_entropy, dist_rep = self.teacher[0].actor.distill_evaluate_actions(obs,
                                                                                        rnn_states_teacher,
                                                                                        actions,
                                                                                        masks,
                                                                                        available_actions,
                                                                                        active_masks)
            self.teacher_action_log_probs = action_log_probs.resize(self.episode_length, self.n_rollout_threads, self.num_agents, 1).to("cpu").numpy()
            self.teacher_dist_rep = dist_rep.resize(self.episode_length,  self.n_rollout_threads, self.num_agents, self.dist_rep_shape).to("cpu").numpy()
        
        torch.set_grad_enabled(torch_grad)

    def feed_forward_generator(self, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        teacher_action_log_probs = self.teacher_action_log_probs.reshape(-1, self.teacher_action_log_probs.shape[-1])
        teacher_dist_rep = self.teacher_dist_rep.reshape(-1, self.teacher_dist_rep.shape[-1])
        # advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            # if advantages is None:
            adv_targ = None
            # else:
            #     adv_targ = advantages[indices]
            teacher_dist_rep_batch = teacher_dist_rep[indices]
            teacher_action_log_probs_batch = teacher_action_log_probs[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch, teacher_dist_rep_batch, teacher_action_log_probs_batch

    def recurrent_generator(self, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        # advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(-1,
                                                                                         *self.rnn_states_critic.shape[
                                                                                          3:])
        teacher_action_log_probs =  _cast(self.teacher_action_log_probs) #.reshape(-1, self.teacher_action_log_probs.shape[-1])
        teacher_dist_rep = _cast(self.teacher_dist_rep) #.reshape(-1, self.teacher_dist_rep.shape[-1])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            # adv_targ = []
            teacher_action_log_probs_batch = []
            teacher_dist_rep_batch = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                # adv_targ.append(advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
                teacher_action_log_probs_batch.append(teacher_action_log_probs[ind:ind + data_chunk_length])
                teacher_dist_rep_batch.append(teacher_dist_rep[ind:ind + data_chunk_length])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)           
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            # adv_targ = np.stack(adv_targ, axis=1)
            teacher_dist_rep_batch = np.stack(teacher_dist_rep_batch, axis=1)
            teacher_action_log_probs_batch = np.stack(teacher_action_log_probs_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            # adv_targ = _flatten(L, N, adv_targ)
            adv_targ = None
            teacher_action_log_probs_batch = _flatten(L, N, teacher_action_log_probs_batch)
            teacher_dist_rep_batch = _flatten(L,N, teacher_dist_rep_batch)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch, teacher_dist_rep_batch, teacher_action_log_probs_batch

    def compute_returns_original(self, next_value):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.shape[0])):
            self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def compute_returns(self):
        for step in reversed(list(range(self.episode_length-1))):
            self.returns[step] = self.returns[step + 1] + self.teacher_action_log_probs[step + 1]

class Trainer(object):
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.args = args
        self.value_normalizer = None

        self.clip_param = args.clip_param
        self.distill_epoch = args.distill_epoch
        self.num_mini_batch = args.num_mini_batch
        self.max_grad_norm = args.max_grad_norm  
        self.entropy_coef = args.entropy_coef     

        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length
    
    def update(self, sample):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, teacher_dist_rep_batch, teacher_action_log_probs_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        teacher_action_log_probs_batch = check(teacher_action_log_probs_batch).to(**self.tpdv)
        teacher_dist_rep_batch = check(teacher_dist_rep_batch).to(**self.tpdv)

        # action_log_probs, dist_entropy, teacher_dist_rep_batch = self.buffer.teacher[0].actor.distill_evaluate_actions(obs_batch, 
        #                                                                       rnn_states_batch, 
        #                                                                       actions_batch, 
        #                                                                       masks_batch, 
        #                                                                         available_actions_batch,
        #                                                                       active_masks_batch)

        # Reshape to do in a single forward pass for all steps
        # student
        action_log_probs, dist_entropy, student_dist_rep = self.policy.actor.distill_evaluate_actions(obs_batch, 
                                                                              rnn_states_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                            active_masks_batch)

        KL_loss = nn.KLDivLoss(reduction="batchmean")
        # def KL_loss_discrete(region_prob1, region_prob2):
        #     return - torch.sum(region_prob1 * region_prob2.log(), dim=-1, keepdim=True)
        
        eps=1e-17

        policy_action_loss = -((action_log_probs * return_batch) * active_masks_batch).sum() / active_masks_batch.sum()
        cross_entropy_loss = (KL_loss((student_dist_rep+eps).log(), (teacher_dist_rep_batch+eps)) * active_masks_batch).sum() / active_masks_batch.sum()
        loss = cross_entropy_loss

        self.policy.actor_optimizer.zero_grad()

        loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        return loss, policy_action_loss, cross_entropy_loss, dist_entropy, actor_grad_norm

    def train(self, buffer, turn_on=True):
        train_info = {}
        self.buffer = buffer

        train_info['loss'] = 0
        train_info['policy_loss'] = 0
        train_info['cross_entropy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0

        for _ in range(self.distill_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(self.num_mini_batch, self.data_chunk_length)
            else:
                data_generator = buffer.feed_forward_generator(num_mini_batch=self.num_mini_batch)

            for sample in data_generator:
    
                loss, policy_loss, cross_entropy_loss, dist_entropy, actor_grad_norm\
                    = self.update(sample)

                train_info['loss'] += loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['cross_entropy_loss'] += cross_entropy_loss.item()

                if int(torch.__version__[2]) < 5:
                    train_info['actor_grad_norm'] += actor_grad_norm
                else:
                    train_info['actor_grad_norm'] += actor_grad_norm.item()

        num_updates = self.distill_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info
    
    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
