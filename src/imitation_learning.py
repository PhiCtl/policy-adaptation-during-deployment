from turtle import forward
import torch
import os
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from copy import deepcopy

from video import VideoRecorder
from arguments import parse_args
from agent.agent import make_agent
from agent.IL_agent import make_il_agent
import utils
from eval import init_env, evaluate
from logger import Logger

"""
This script trains Imitation learning agents from RL experts trained on different domains (different by their dynamics
parameters, either the mass or the force).
Those IL agents have a shared domain generic module, which has the same architecture as the original "PAD" agent we've
explored so far. They have their own domain specific module, which takes as input the groundtruth (ie. actual)
dynamics value, eg. the mass of the system. The output of this module is concatenated with the output from
shared encoder, as input to SS and actor heads
"""

def evaluate_agent(agent, env, args, feat_analysis=False, step=None, L=None): # OK
    """Evaluate agent on env, storing obses, actions and next obses
       if feat_analysis, then we also return the feature vector (output of the domain
       specific module)"""

    ep_rewards = []
    obses, actions, feat_vects = [], [], []

    for i in range(args.num_rollouts):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        rewards = []

        agent.train()

        while not done:

            # Take a step
            mass = env.get_masses() # TODO change to env.get_forces()
            with utils.eval_mode(agent):
                action = agent.select_action(obs, mass)
            next_obs, reward, done, info, _, _ = env.step(action, rewards)
            episode_reward += reward
            if L and step:
                L.log('eval/episode_reward', episode_reward, step)
            obses.append(obs)
            actions.append(action)
            #feat_vects.append(agent.extract_feat_vect(mass)) # You can comment it if it raises errors but it should not
            obs = next_obs
            step += 1

        obses.append(obs)
        ep_rewards.append(episode_reward)

    if L and step:
        L.dump(step)

    if feat_analysis:
        return np.array(ep_rewards), obses, actions, feat_vects
    return np.array(ep_rewards), obses, actions


def collect_expert_samples(agent, env, args, label): # OK
    """Collect samples for Imitation learning training
       Args : - agent : expert RL agent trained on env
              - env : dm_control environment
              - args
              - label : env specificity, eg. the cartmass
       """
    # Create replay buffer with label
    buffer = utils.SimpleBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.train_steps,
        batch_size=args.batch_size,
        label=label
    )

    ep_rewards, obses, actions = evaluate_agent(agent, env, args)
    buffer.add_path(obses, actions) # Save policy rollout
    return buffer, ep_rewards.mean(), ep_rewards.std()

def relabel(obses, expert): # OK
    """Relabel observations with expert agent according to DAgger algorithm"""
    with utils.eval_mode(expert):
        actions_new = []
        for obs in obses:
            actions_new.append(expert.select_action(obs))
    return actions_new

def load_agent(label, action_shape, args): # OK
    """Load rL expert model from directory"""

    work_dir = args.work_dir + label # example : logs/cartpole_swingup + "_0_3"
    L = Logger(work_dir, use_tb=True, config='il')
    model_dir = os.path.join(work_dir, 'inv', '0', 'model') # logs/cartpole_swingup_0_3/inv/0/model
    print(f'Load agent from {work_dir}')

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=action_shape,
        args=args
    )
    agent.load(model_dir, args.pad_checkpoint)

    return agent, L
    
def main(args):


    # TODO better practise than lists
    labels = ["_0_4", "_0_2", "_0_25", "_0_3"] # TODO change labels with forces directory labels

    # 1. Define 4 envts
    print("-"*60)
    print("Define environment")
    envs = []
    masses = [] # TODO change to forces
    for mass in [0.4, 0.2, 0.25, 0.3]:
        env = init_env(args, mass)
        masses.append(env.get_masses()) # TODO env.get_forces()
        envs.append(env)

    # 2. Load expert agents
    print("-" * 60)
    print("Load experts")
    experts = []
    loggers = []
    for label in labels:
        # All envs have should have the same action space shape
        agent, logger = load_agent(label, envs[0].action_space.shape, args)
        experts.append(agent)
        loggers.append(logger)

    # 3. Collect samples from 4 RL agents
    print("-" * 60)
    print("Fill in buffers")
    buffers = [] # save data for IL
    stats_expert = dict() # save score of trained RL agents on corresponding environments
    stats_il = {k:[] for k in labels} # save score of Il agents

    # Initialize buffers by collecting experts data and collect their performance in the meantime
    for expert, mass, env in zip(experts, labels, envs) : # TODO change mass to forces
        buffer, mean, std = collect_expert_samples(expert, env, args, mass)
        buffers.append(buffer)
        stats_expert[mass] = [mean, std]

    # 4. Create IL agents
    print("-" * 60)
    print("Create IL agents")
    il_agents = []
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)

    for mass in masses: # TODO replace with forces
        il_agent = make_il_agent(
            obs_shape=cropped_obs_shape,
            action_shape=envs[0].action_space.shape,
            dynamics_input_shape=mass.shape[0], # replace with force
            args=args)
        il_agents.append(il_agent)

    # Share domain generic part between agents
    il_agents_tied = [il_agents[0]]
    for il_agent in il_agents[1:]:
        il_agent.tie_agent_from(il_agents[0])
        il_agents_tied.append(il_agent)

    # 5. Train the four IL agents with DAgger algorithm
    print("-" * 60)
    print("Train IL agents")

    for it in range(args.n_iter): # number of dagger iterations
        print("\n\n********** Training %i ************"%it)

        # Train 4 Il agents policies
        for step in range(args.il_steps):

            # Save data
            preds, pred_invs, gts, losses = [], [], [], 0

            # Forward pass sequentially for all agents
            for agent, buffer, mass, L in zip(il_agents_tied, buffers, masses, loggers): # TODO change to forces
                obs, action, next_obs, _ = buffer.sample() # sample a batch
                action_pred, action_inv, loss = agent.predict_action(obs, next_obs, mass, action, L=L, step=step)

                preds.append(action_pred) # Action from actor network
                pred_invs.append(action_inv) # Action from SS head
                gts.append(action)
                losses += loss

            # Backward pass
            losses.backward()

            for agent in il_agents_tied:
                agent.update()

        # Evaluate - Perform IL agent policy rollouts
        print("\n\n********** Evaluation and relabeling %i ************" % it)
        for agent, expert, logger, env, buffer, mass in zip(il_agents_tied, experts, loggers, envs, buffers, labels):
            rewards, obses, actions = evaluate_agent(agent, env, args, L=logger, step=step) # evaluate agent on environment
            stats_il[mass].append([rewards.mean(), rewards.std()]) # save intermediary score
            print(f'Performance of agent on mass {mass} : {rewards.mean()} +/- {rewards.std()}')
            actions_new = relabel(obses, expert)
            buffer.add_path(obses, actions_new)


        # Save partial model
        if it % 5 == 0 :
            for agent, label in zip(il_agents_tied, labels):
                save_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
                agent.save(save_dir, it)


    # 6. Save IL agents
    for agent, label in zip(il_agents_tied, labels):
        save_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
        agent.save(save_dir, "final")


    # 7. Evaluate expert vs IL
    for label in labels :
        print("-"*60)
        print(f'Mass of {label}')
        print(f'Expert performance : {stats_expert[label][0]} +/- {stats_expert[label][1]}')
        print(f'Imitation learning agent with dagger performance : {stats_il[label][-1][0]} +/- {stats_il[label][-1][1]}')


if __name__ == '__main__':
    args = parse_args()
    main(args)