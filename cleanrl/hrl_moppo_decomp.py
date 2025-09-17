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
from cleanrl_utils import get_base_env
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics, SingleRewardWrapper
from gymnasium.wrappers import TimeLimit
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from cleanrl.moppo_decomp import Agent

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
    total_timesteps: int = 1e7#1e5
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4 #4
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
    ent_coef: float = 0.05 #0.01
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

    checkpoint_interval: int = 100
    """the checkpoint interval in iterations"""
    resume_from: str = ""  
    """the relative checkpoint pt to load checkpoint from"""
    run_name_modifier: str = ""  
    """run name modifier"""

    obj_duration: int = 5
    """how many primitive steps the chosen low-level policy runs for each high-level decision"""

def make_env(env_id, idx, capture_video, run_name):
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
        #env = SingleRewardWrapper(env, obj_idx)

        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
class Controller(nn.Module):
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
            layer_init(nn.Linear(64, get_base_env(envs.envs[0]).reward_dim), std=0.01),#num objectives
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

    # env setup
    envs = MOSyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )

    num_objectives = get_base_env(envs.envs[0]).reward_dim

    base_env = get_base_env(envs.envs[0])
    controller = Controller(envs).to(device)
    agents = [Agent(envs).to(device) for i in range(num_objectives)]

    checkpoint_path = "../model/four-room-easy/checkpoint_4880.pt"  # adjust path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for idx, agent in enumerate(agents):
        agent.load_state_dict(checkpoint["agents"][idx])
    
    #TODO make param for freezing agents
    for agent in agents:
        agent.eval()
        for p in agent.parameters():
            p.requires_grad = False
    optimizer = optim.Adam(
        controller.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
    )

    # HIGH-LEVEL storage (for controller decisions)
    # shapes: (num_steps, num_envs, *obs_shape) and (num_steps, num_envs) for scalars
    obs_shape = envs.single_observation_space.shape
    obs_h = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions_h = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs_h = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_h = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_h = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_h = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        controller.load_state_dict(checkpoint["controller"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = checkpoint["iteration"] + 1
        global_step = checkpoint["global_step"]
        print(f"Resumed from {args.resume_from} at iteration {start_iteration}")
    else:
        start_iteration = 1

    for iteration in range(start_iteration, int(args.num_iterations) + 1): #So each iteration is a full cycle of data collection + training.
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # max_episode_steps = 500
        # initialize before step loop
        running_returns = [np.zeros(num_objectives) for _ in range(args.num_envs)]  #num_objectives lists of size num_envs
        running_lengths = np.zeros(args.num_envs)

        #GENERATING EXPERIENCE
        gamma_h = args.gamma ** args.obj_duration
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_h[step] = next_obs.clone()

            with torch.no_grad():
                # flatten obs for network if needed (controller expects flat vector)
                flat_obs = next_obs.view(args.num_envs, -1)
                hl_action, hl_logprob, hl_entropy, hl_value = controller.get_action_and_value(flat_obs)
                # hl_action: tensor shape (num_envs,), values in {0..num_objectives-1}
                high_actions = hl_action.detach()
                logprobs_h[step] = hl_logprob.detach()
                values_h[step] = hl_value.detach().view(-1)

            # We'll run the chosen low-level policy for obj_duration primitive steps.
            # For each primitive step we must produce a full action vector for the vector env.
            cum_reward = np.zeros(args.num_envs, dtype=np.float32)  # discounted-sum per env

            # ensure numpy versions for mask indexing
            high_actions_np = high_actions.cpu().numpy()
            for i, env_action in enumerate(high_actions):
                writer.add_scalar(f"charts/chosen_obj_env{i}", env_action, global_step)   

            for k in range(args.obj_duration): #each env is seperate
                # For each distinct selected objective, compute actions for envs that chose it
                # Build actions array for all envs
                actions_batch = np.zeros((args.num_envs,), dtype=np.int64)
                for opt_idx in range(num_objectives):
                    mask = np.where(high_actions_np == opt_idx)[0]
                    if mask.size == 0:
                        continue
                    # gather obs for these env indices
                    obs_group = next_obs[mask]  # shape (n_mask, *obs_shape)
                    flat_obs_group = obs_group.view(len(mask), -1)
                    with torch.no_grad():
                        # low-level agent returns an action per sample in the group
                        action_group, _, _, _ = agents[opt_idx].get_action_and_value(flat_obs_group) #take action with chosen policy
                    actions_batch[mask] = action_group.cpu().numpy().astype(np.int64)
                # step the vector env with the assembled actions
                next_obs_np, reward_np, terminations, truncations, infos = envs.step(actions_batch)
                # reward_np shape is (num_envs, reward_dim)
                # compute scalarised reward per env
                scalar_reward = reward_np.sum(axis=1).astype(np.float32)
                # accumulate discounted sum within the option
                cum_reward += (args.gamma ** k) * scalar_reward
                # update done and obs for the next primitive step
                finished = np.logical_or(terminations, truncations)
                # convert next_obs_np to tensor for next iteration
                next_obs = torch.Tensor(next_obs_np).to(device)
                next_done = torch.Tensor(finished.astype(np.float32)).to(device)

                running_returns += reward_np
                running_lengths += 1

                # reset running sums for finished envs
                finished_returns = running_returns[finished]
                finished_lengths = running_lengths[finished]
                running_returns[finished] = 0.0
                running_lengths[finished] = 0
                # (Optional) handle logging per env for finished episodes if desired
                # We'll aggregate episodic returns via MORecordEpisodeStatistics wrapper as before

                if finished_returns.size > 0:
                    writer.add_scalar(f"charts/mean_episodic_length", finished_lengths.mean(), global_step)        
                    for env_returns in finished_returns:
                        for i, obj_return in enumerate(env_returns):
                            writer.add_scalar(f"charts/episodic_return_obj{i}", obj_return, global_step)
                        writer.add_scalar(f"charts/mean_obj_episodic_return", env_returns.mean().astype(np.float32), global_step)
                # If all envs are done, break early
                if finished.all():
                    break
                


            # store high-level reward and done flags for this high-level decision
            rewards_h[step] = torch.tensor(cum_reward, dtype=torch.float32).to(device)
            dones_h[step] = next_done

            # store high-level action
            actions_h[step] = high_actions.long()
            #for i in range(len(cum_reward)):
            #    writer.add_scalar(f"charts/cum_reward_opt_env{i}", cum_reward[i], global_step)
            writer.add_scalar(f"charts/avg_cum_reward_opt_envs", np.average(cum_reward), global_step)
        #CALCULATING BEFORE TRAINING
        # bootstrap value if not done
        # --- compute GAE and returns for high-level agent ---
        advantages = torch.zeros_like(rewards_h).to(device)
        returns = torch.zeros_like(rewards_h).to(device)
        with torch.no_grad():
            next_value = controller.get_value(next_obs.view(args.num_envs, -1)).reshape(1, -1)  # shape (1, num_envs)

        lastgaelam = torch.zeros(args.num_envs).to(device)
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextval = next_value.view(-1)
            else:
                nextnonterminal = 1.0 - dones_h[t + 1]
                nextval = values_h[t + 1]

            delta = rewards_h[t] + gamma_h * nextval * nextnonterminal - values_h[t]
            advantages[t] = lastgaelam = delta + gamma_h * args.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + values_h

        # TRAINING
        # flatten the batch
            # Flatten batch for PPO update
        obs = obs_h.reshape((-1,) + obs_shape)
        actions = actions_h.reshape(-1)
        logprobs = logprobs_h.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        values_flat = values_h.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        pg_loss = v_loss = entropy_loss = torch.tensor(0.0)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = controller.get_action_and_value(
                    obs[mb_inds].view(len(mb_inds), -1), actions.long()[mb_inds]
                )
                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = advantages_flat[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_batch = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - returns_flat[mb_inds]) ** 2
                    v_clipped = values_flat[mb_inds] + torch.clamp(newvalue - values_flat[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - returns_flat[mb_inds]) ** 2
                    v_loss_batch = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss_batch = 0.5 * ((newvalue - returns_flat[mb_inds]) ** 2).mean()

                entropy_batch = entropy.mean()
                loss = pg_loss_batch - args.ent_coef * entropy_batch + v_loss_batch * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(controller.parameters(), args.max_grad_norm)
                optimizer.step()

                pg_loss += pg_loss_batch.item()
                v_loss += v_loss_batch.item()
                entropy_loss += entropy_batch.item()

        # Logging
        writer.add_scalar("losses/value_loss", v_loss / (args.update_epochs * (args.batch_size / args.minibatch_size)), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss / (args.update_epochs * (args.batch_size / args.minibatch_size)), global_step)
        writer.add_scalar("losses/entropy", entropy_loss / (args.update_epochs * (args.batch_size / args.minibatch_size)), global_step)
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        if iteration % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt")
            torch.save({
                "iteration": iteration,
                "global_step": global_step,
                "controller": controller.state_dict(),
                "obj_duration": args.obj_duration,
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    envs.close()
    writer.close()
