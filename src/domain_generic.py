from turtle import forward
import torch
import os
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from copy import deepcopy

from video import VideoRecorder
from utils_imitation_learning import *
from arguments import parse_args
from agent.agent import make_agent
from agent.agent_domain_generic import make_domain_generic_agent
import utils
from eval import init_env, evaluate
from logger import Logger

def evaluate_agent(agent, env, args): # OK
    """Evaluate agent on env, storing obses, actions and next obses in buffer if any"""

    ep_rewards = []
    obses, actions = [], []
    
    for i in range(args.num_rollouts):
        obs = env.reset()
        done = False
        
        episode_reward, step, rewards = 0, 0, []

        agent.train()

        while not done:

            # Take a step
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            next_obs, reward, done, info, change, _ = env.step(action, rewards)
            episode_reward += reward
            obses.append(obs)
            actions.append(action)

            obs = next_obs
            step += 1
            

        obses.append(obs)
        ep_rewards.append(episode_reward)

    return np.array(ep_rewards), obses, actions

# def collect_expert_samples(agent, env, args, label = None): # OK
#     """Collect samples for Imitation learning training
#        Args : - agent : expert RL agent trained on env
#               - env : dm_control environment
#               - args
#               - label : env specificity, eg. the cartmass
#        """
#     # Create replay buffer with label
#     buffer = utils.SimpleBuffer(
#         obs_shape=env.observation_space.shape,
#         action_shape=env.action_space.shape,
#         capacity=args.train_steps,
#         batch_size=args.batch_size,
#         label=label
#     )

#     ep_rewards, obses, actions = evaluate_agent(agent, env, args)
#     buffer.add_path(obses, actions)
#     return buffer, ep_rewards.mean(), ep_rewards.std()

# def relabel(obses, expert): # OK
#     """Relabel observations with expert agent"""
#     with utils.eval_mode(expert):
#         actions_new = []
#         for obs in obses:
#             actions_new.append(expert.select_action(obs))
#     return actions_new

# def load_agent(label, action_shape, args): # OK
#     """Load rL expert model from directory"""

#     work_dir = args.work_dir + label # example : logs/cartpole_swingup + "_0_3"
#     L = Logger(work_dir, use_tb=True, config='il')
#     model_dir = os.path.join(work_dir, 'inv', '0', 'model') # logs/cartpole_swingup_0_3/inv/0/model
#     print(f'Load agent from {work_dir}')

#     # Prepare agent
#     assert torch.cuda.is_available(), 'must have cuda enabled'
#     cropped_obs_shape = (3 * args.frame_stack, 84, 84)
#     agent = make_agent(
#         obs_shape=cropped_obs_shape,
#         action_shape=action_shape,
#         args=args
#     )
#     agent.load(model_dir, args.pad_checkpoint)

#     return agent, L

def main(args):
    labels = ["_0_4", "_0_2", "_0_25", "_0_3"]  # for cart-pole mass
    # labels = ["_0_-1", "_0_-2", "_0_-3"] #for walker-walk force
    domains = [0.4, 0.2, 0.25, 0.3]  # for cart-pole mass
    # domains = [-1, -2, -3] #for walker-walk force
    il_agents, experts, envs, dynamics, buffers, _, stats_expert = setup(args,
                                                                         labels=labels,
                                                                         domains=domains,
                                                                         gt=True,
                                                                         train_IL=True,
                                                                         type="mass")
    il_agent = il_agents[0]

    stats_il = {k: [] for k in labels}  # save score of Il agents

    # Train the domains IL agents with DAgger algorithm
    print("-" * 60)
    print("Train IL agents")

    for it in tqdm(range(args.n_iter)):  # number of dagger iterations
        print("\n\n********** Training %i ************" % it)

        # Train domains Il agents policies
        for step in tqdm(range(args.il_steps)):

            # Save data
            preds, pred_invs, gts, losses = [], [], [], 0

            # Forward pass sequentially for all agents
            for buffer, mass in zip(buffers, dynamics):
                obs, action, next_obs = buffer.sample()  # sample a batch
                action_pred, action_inv, loss = il_agent.predict_action(obs, next_obs, action, mass=mass)

                preds.append(action_pred)  # Action from actor network
                pred_invs.append(action_inv)  # Action from SS head
                gts.append(action)
                losses += loss

            # Backward pass
            losses.backward()
            if step % 1000 == 0: print(losses)

            il_agent.update()

        # Evaluate - Perform IL agent policy rollouts
        print("\n\n********** Evaluation and relabeling %i ************" % it)
        for expert, env, buffer, mass in zip(experts, envs, buffers, labels):
            rewards, obses, actions = evaluate_agent(il_agent, env, args)  # evaluate agent on environment
            stats_il[mass].append([rewards.mean(), rewards.std()])  # save intermediary score
            print(f'Performance of agent on force {mass} : {rewards.mean()} +/- {rewards.std()}')
            actions_new = relabel(obses, expert)
            buffer.add_path(obses, actions_new)

        # Save partial model
        if it % 2 == 0:

            save_dir = utils.make_dir(os.path.join(args.save_dir, "_domain_generic", 'model'))
            il_agent.save(save_dir, it)

    # 6. Save IL agents
    save_dir = utils.make_dir(os.path.join(args.save_dir, "_domain_generic", 'model'))
    il_agent.save(save_dir, it)


    # 7. Evaluate expert vs IL
    for label in labels:
        print("-" * 60)
        print(f'Mass of {label}')
        print(f'Expert performance : {stats_expert[label][0]} +/- {stats_expert[label][1]}')
        print(
            f'Imitation learning agent with dagger performance : {stats_il[label][-1][0]} +/- {stats_il[label][-1][1]}')

if __name__ == '__main__':
    args = parse_args()
    main(args)