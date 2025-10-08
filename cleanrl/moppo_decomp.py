# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics, SingleRewardWrapper
from gymnasium.wrappers import TimeLimit
from cleanrl_utils.utils import get_base_env


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "four-room-easy-v0"
    """the id of the environment"""
    total_timesteps: int = 1e7  # 1e5
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 2  # 4
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.05  # 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    checkpoint_interval: int = 10
    """the checkpoint interval in iterations"""
    resume_from: str = ""
    """the relative checkpoint pt to load checkpoint from"""
    run_name_modifier: str = ""
    """run name modifier"""


def make_env(env_id, obj_idx, capture_video, run_name):
    def thunk():
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array")
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        #     env = gym.make(env_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        env = mo_gym.make(env_id)
        # env = mo_gym.wrappers.LinearReward(env, weight=np.array([0.8, 0.2]))
        # env = TimeLimit(env, max_episode_steps=100)  # ensure episodes end
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        env = SingleRewardWrapper(env, obj_idx)

        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.run_name_modifier}__{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    checkpoint_dir = f"checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # create a single environment to inspect the reward dimension
    temp_env = mo_gym.make(args.env_id)
    temp_env = MORecordEpisodeStatistics(temp_env, gamma=0.98)
    num_objectives = temp_env.reward_dim
    temp_env.close()

    # env setup
    envs_list = [
        gym.vector.SyncVectorEnv(
            [
                make_env(args.env_id, obj_idx, args.capture_video, run_name)
                for _ in range(args.num_envs)
            ]
        )
        for obj_idx in range(num_objectives)
    ]

    agents = [Agent(envs_list[i]).to(device) for i in range(num_objectives)]

    optimizers = [
        optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        for agent in agents
    ]

    # ALGO Logic: Storage setup
    # storage: one per agent
    obs_list = [
        torch.zeros(
            (args.num_steps, args.num_envs)
            + envs_list[i].single_observation_space.shape
        ).to(device)
        for i in range(num_objectives)
    ]
    actions_list = [
        torch.zeros(
            (args.num_steps, args.num_envs) + envs_list[i].single_action_space.shape
        ).to(device)
        for i in range(num_objectives)
    ]
    logprobs_list = [
        torch.zeros((args.num_steps, args.num_envs)).to(device)
        for _ in range(num_objectives)
    ]
    rewards_list = [
        torch.zeros((args.num_steps, args.num_envs)).to(device)
        for _ in range(num_objectives)
    ]
    values_list = [
        torch.zeros((args.num_steps, args.num_envs)).to(device)
        for _ in range(num_objectives)
    ]
    dones_list = [
        torch.zeros((args.num_steps, args.num_envs)).to(device)
        for _ in range(num_objectives)
    ]

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # reset all envs
    next_obs_list = []
    next_done_list = []

    for obj_idx, vec_env in enumerate(envs_list):
        # reset vectorised env
        obs, _ = vec_env.reset(seed=args.seed)

        # get specialised obs for each underlying env
        spec_obs_list = []
        for env in vec_env.envs:
            base_env = get_base_env(env)
            spec_obs = base_env.update_specialisation(obj_idx + 1)  # returns masked obs
            spec_obs = spec_obs.squeeze(0)  # now [obs_dim]
            spec_obs_list.append(spec_obs)

        # Stack all envs into a single batch tensor
        spec_obs_batch = np.concatenate(spec_obs_list, axis=0)  # merges along first dim
        next_obs_list.append(torch.Tensor(spec_obs_batch).to(device))
        next_done_list.append(
            torch.zeros(args.num_envs, dtype=torch.float32).to(device)
        )

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        for agent, state_dict in zip(agents, checkpoint["agents"]):
            agent.load_state_dict(state_dict)
        for optimizer, state_dict in zip(optimizers, checkpoint["optimizers"]):
            optimizer.load_state_dict(state_dict)
        start_iteration = checkpoint["iteration"] + 1
        global_step = checkpoint["global_step"]
        print(f"Resumed from {args.resume_from} at iteration {start_iteration}")
    else:
        start_iteration = 1

    for iteration in range(
        start_iteration, int(args.num_iterations) + 1
    ):  # So each iteration is a full cycle of data collection + training.
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for i, opt in enumerate(optimizers):
                opt.param_groups[0]["lr"] = lrnow
                
        # max_episode_steps = 500
        # initialize before step loop
        running_returns = [
            np.zeros(args.num_envs) for _ in range(num_objectives)
        ]  # num_objectives lists of size num_envs
        running_lengths = [np.zeros(args.num_envs) for _ in range(num_objectives)]

        # GENERATING EXPERIENCE
        for step in range(0, args.num_steps):
            mean_returns_per_obj = []
            mean_lengths_per_obj = []
            for i, envs in enumerate(
                envs_list
            ):  # EACH IS A VEC ENV i.e. loop each objective seperately
                global_step += args.num_envs
                agent = agents[i]
                next_obs = next_obs_list[i]
                next_done = next_done_list[i]

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values_list[i][step] = value.flatten()

                actions_list[i][step] = action
                logprobs_list[i][step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                # step the environment
                next_obs_batch, reward_batch, terminations, truncations, infos = (
                    envs.step(action.cpu().numpy())
                )

                rewards_list[i][step] = torch.tensor(reward_batch).to(device).view(-1)
                next_done_list[i] = torch.tensor(
                    np.logical_or(terminations, truncations)
                ).to(device)
                next_obs_list[i] = torch.Tensor(next_obs_batch).to(device)

                running_returns[i] += reward_batch
                running_lengths[i] += 1

                finished_mask = np.logical_or(terminations, truncations)
                # reset running sums for finished envs
                finished_returns = running_returns[i][finished_mask]
                finished_lengths = running_lengths[i][finished_mask]
                running_returns[i][finished_mask] = 0.0
                running_lengths[i][finished_mask] = 0

                if finished_returns.size > 0:
                    mean_return_per_vec = finished_returns.mean()
                    mean_length_per_vec = finished_lengths.mean()
                    writer.add_scalar(
                        f"charts/mean_episodic_return_obj{i}",
                        mean_return_per_vec,
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts/mean_episodic_length_obj{i}",
                        mean_length_per_vec,
                        global_step,
                    )
                    mean_returns_per_obj.append(mean_return_per_vec)
                    mean_lengths_per_obj.append(mean_length_per_vec)
            # compute mean across all objectives
            if len(mean_returns_per_obj) > 0:
                overall_mean_return = np.mean(mean_returns_per_obj)
                overall_mean_length = np.mean(mean_lengths_per_obj)
                writer.add_scalar(
                    "charts/mean_episodic_return_all_objs",
                    overall_mean_return,
                    global_step,
                )
                writer.add_scalar(
                    "charts/mean_episodic_length_all_objs",
                    overall_mean_length,
                    global_step,
                )

        # CALCULATING BEFORE TRAINING
        # bootstrap value if not done
        advantages_list = []
        returns_list = []
        with torch.no_grad():
            next_values_list = [
                agents[i].get_value(next_obs_list[i]).reshape(1, -1)
                for i in range(num_objectives)
            ]

            for i in range(num_objectives):
                rewards = rewards_list[i]
                values = values_list[i]
                dones = dones_list[i]
                next_values = next_values_list[i]

                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done_list[i].float()
                        nextval = next_values
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextval = values[t + 1]

                    delta = (
                        rewards[t] + args.gamma * nextval * nextnonterminal - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )

                returns = advantages + values
                advantages_list.append(advantages)
                returns_list.append(returns)

        v_loss_list = []
        pg_loss_list = []
        entropy_loss_list = []
        old_approx_kl_list = []
        approx_kl_list = []
        clipfracs_list = []

        # TRAINING
        # flatten the batch
        for i, agent in enumerate(
            agents
        ):  # TRAIN EACH AGENT ON ITS CORRESPONDING EXPERIENCE
            obs = obs_list[i].reshape(
                (-1,) + envs_list[i].single_observation_space.shape
            )
            actions = actions_list[i].reshape(
                (-1,) + envs_list[i].single_action_space.shape
            )
            logprobs = logprobs_list[i].reshape(-1)
            advantages = advantages_list[i].reshape(-1)
            returns = returns_list[i].reshape(-1)
            values = values_list[i].reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        obs[mb_inds], actions.long()[mb_inds]
                    )
                    logratio = newlogprob - logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                        v_clipped = values[mb_inds] + torch.clamp(
                            newvalue - values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    optimizers[i].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizers[i].step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            writer.add_scalar(f"losses/value_loss_obj{i}", v_loss.item(), global_step)
            writer.add_scalar(f"losses/policy_loss_obj{i}", pg_loss.item(), global_step)
            writer.add_scalar(
                f"losses/entropy_obj{i}", entropy_loss.item(), global_step
            )
            writer.add_scalar(
                f"losses/old_approx_kl_obj{i}", old_approx_kl.item(), global_step
            )
            writer.add_scalar(f"losses/approx_kl_obj{i}", approx_kl.item(), global_step)
            writer.add_scalar(
                f"losses/clipfrac_obj{i}", np.mean(clipfracs), global_step
            )

            v_loss_list.append(v_loss.item())
            pg_loss_list.append(pg_loss.item())
            entropy_loss_list.append(entropy_loss.item())
            old_approx_kl_list.append(old_approx_kl.item())
            approx_kl_list.append(approx_kl.item())
            clipfracs_list.append(np.mean(clipfracs))

            y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizers[obj_idx].param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", np.mean(v_loss_list), global_step)
        writer.add_scalar("losses/policy_loss", np.mean(pg_loss_list), global_step)
        writer.add_scalar("losses/entropy", np.mean(entropy_loss_list), global_step)
        writer.add_scalar(
            "losses/old_approx_kl", np.mean(old_approx_kl_list), global_step
        )
        writer.add_scalar("losses/approx_kl", np.mean(approx_kl_list), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs_list), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        if iteration % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt")
            torch.save(
                {
                    "iteration": iteration,
                    "global_step": global_step,
                    "agents": [agent.state_dict() for agent in agents],
                    "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    for envs in envs_list:
        envs.close()
    writer.close()
