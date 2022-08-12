import torch
from torch import nn
import torch.nn.functional as F
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
import numpy as np

def get_dist_rep_shape(args):
    '''
    return an integer, the dims of repre for action distributions
    '''

    return 1

    # return args.grid_size ** 2 + 2 * 2

def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols

class DistillationBuffer(SharedReplayBuffer):
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        super(DistillationBuffer, self).__init__(args, num_agents, obs_space, share_obs_space, act_space)
        dist_rep_shape = get_dist_rep_shape(args)
        self.teacher_dist_rep = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, dist_rep_shape), dtype=np.float32)
        self.teacher_action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, dist_rep_shape), dtype=np.float32)
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

        # for MAT

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

        _, action_log_probs, dist_entropy = self.teacher[0].evaluate_actions(share_obs, 
                                                                                obs, 
                                                                                rnn_states, 
                                                                                rnn_states,
                                                                                actions,
                                                                                masks, 
                                                                                available_actions,
                                                                                active_masks)
        self.teacher_action_log_probs = action_log_probs.resize(self.episode_length, self.n_rollout_threads, self.num_agents, get_dist_rep_shape(self.args)).to("cpu").numpy()
        self.teacher_dist_rep = dist_entropy.resize(self.episode_length, self.n_rollout_threads, self.num_agents, get_dist_rep_shape(self.args)).to("cpu").numpy()
        
        # for MAPPO
        # for e in range(self.n_rollout_threads):
        #     obs = self.obs[:-1, e].reshape(-1, *self.obs.shape[3:])
        #     rnn_states = self.rnn_states[:-1, e].reshape(-1, *self.rnn_states.shape[3:])
        #     if self.available_actions is not None:
        #         available_actions = self.available_actions[:-1, e].reshape(-1, self.available_actions.shape[-1])
        #     masks = self.masks[:-1, e].reshape(-1, 1)
        #     actions = self.actions[:, e].reshape(-1, self.actions.shape[-1])
        #     active_masks = self.active_masks[:-1, e].reshape(-1, 1)

        #     action_log_probs, dist_entropy = self.teacher[e].actor.evaluate_actions(obs,
        #                                                                             rnn_states,
        #                                                                             actions,
        #                                                                             masks,
        #                                                                             available_actions,
        #                                                                             active_masks)
        #     self.teacher_action_log_probs[:, e, :, :] = action_log_probs.resize(self.episode_length, self.num_agents, get_dist_rep_shape(self.args)).to("cpu").numpy()

        # share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        # obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        # rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        # rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        # actions = self.actions.reshape(-1, self.actions.shape[-1])
        # if self.available_actions is not None:
        #     available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        # value_preds = self.value_preds[:-1].reshape(-1, 1)
        # returns = self.returns[:-1].reshape(-1, 1)
        # masks = self.masks[:-1].reshape(-1, 1)
        # active_masks = self.active_masks[:-1].reshape(-1, 1)

        # action_log_probs, dist_entropy = self.teacher[0].actor.evaluate_actions(obs,
        #                                                                             rnn_states,
        #                                                                             actions,
        #                                                                             masks,
        #                                                                             available_actions,
        #                                                                             active_masks)
        # self.teacher_action_log_probs = action_log_probs.resize(self.episode_length, self.n_rollout_threads, self.num_agents, get_dist_rep_shape(self.args)).to("cpu").numpy()

        # print(action_log_probs)
        # action_log_probs: torch.Size([8000, 1])
        # self.teacher_dist_rep[:, e, :, :] = dist_rep.resize(self.episode_length, self.num_agents, get_dist_rep_shape(self.args)).to("cpu").numpy()
        
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
        teacher_dist_rep = self.teacher_dist_rep.reshape(-1, self.teacher_action_log_probs.shape[-1])
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

    def compute_returns(self):
        # original: 
        # self.returns[-1] = next_value
        # for step in reversed(range(self.rewards.shape[0])):
        #     self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]
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

        # Reshape to do in a single forward pass for all steps
        # student
        action_log_probs, dist_entropy = self.policy.actor.evaluate_actions(obs_batch, 
                                                                              rnn_states_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        CEL = nn.MSELoss()
        if self._use_policy_active_masks:
            policy_action_loss = -((action_log_probs * return_batch) * active_masks_batch).sum() / active_masks_batch.sum()
            # print(CEL(action_log_probs, torch.tensor(teacher_action_log_probs_batch).to(self.device)))
            cross_entropy_loss = (CEL(dist_entropy, torch.tensor(teacher_dist_rep_batch).to(self.device)) * active_masks_batch).sum() / active_masks_batch.sum()
            # cross_entropy_loss = (self.compute_dist_cross_entropy(student_dist_rep, teacher_dist_rep_batch) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -(action_log_probs * return_batch).mean()
            cross_entropy_loss = CEL(dist_entropy, torch.tensor(teacher_dist_rep_batch).to(self.device)).mean()
            # cross_entropy_loss = self.compute_dist_cross_entropy(student_dist_rep, teacher_dist_rep_batch).mean()
        loss = policy_action_loss + cross_entropy_loss

        # # second way for computing loss
        # imp_weights = torch.exp(action_log_probs - teacher_action_log_probs_batch)

        # # surr1 = imp_weights * adv_targ
        # adv_targ = 1
        # surr1 = imp_weights * adv_targ
        # surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        # if self._use_policy_active_masks:
        #     policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
        #                                      dim=-1,
        #                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        # else:
        #     policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # policy_loss = policy_action_loss

        # # policy_action_loss = -action_log_probs.mean()
        # entropy_loss = -dist_entropy *  self.entropy_coef
        # loss = policy_loss + entropy_loss

        self.policy.actor_optimizer.zero_grad()

        loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        return loss, policy_action_loss, cross_entropy_loss, dist_entropy, actor_grad_norm
    
    # def compute_dist_cross_entropy(self, dist1, dist2):
    #     # assert self.args.grid_goal and self.args.use_grid_simple and dist1.shape[-1] == (self.args.grid_size ** 2) + 4
    #     # grid_size ** 2 x Discrete + 2 x Normal (mean + std)
    #     num_grids = self.args.grid_size ** 2
    #     region_prob1, point_mean1, point_std1 = dist1[:, :num_grids], dist1[:, num_grids:num_grids+2], dist1[:, num_grids+2:num_grids+4]
    #     region_prob2, point_mean2, point_std2 = dist2[:, :num_grids], dist2[:, num_grids:num_grids+2], dist2[:, num_grids+2:num_grids+4]
    #     region_v = - torch.sum(region_prob1 * region_prob2.log(), dim=-1, keepdim=True)
    #     point_v = (point_std2 * np.sqrt(2*np.pi)).log() + 0.5 * (point_std1.pow(2) + (point_mean1 - point_mean2).pow(2)) / point_std2.pow(2)
    #     point_v = torch.sum(point_v, dim=-1, keepdim=True)
    #     return region_v + point_v

    def train(self, buffer, turn_on=True):
        train_info = {}
        self.buffer = buffer

        train_info['loss'] = 0
        train_info['policy_loss'] = 0
        train_info['cross_entropy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0

        for _ in range(self.distill_epoch):
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