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
from agent.IL_agent_visual import make_il_agent_visual
import utils
from eval import init_env, evaluate
from logger import Logger

"""
This script trains Imitation learning agents from RL experts trained on different domains (different by their dynamics
parameters, either the mass or the force).
Those IL agents have a shared domain generic module, which has the same architecture as the original "PAD" agent we've
explored so far. They have their own domain specific module, which takes as input the past 3 observations and actions 
(ie. obs, act, obs, act, obs) to infer the dynamics. The output of this module is concatenated with output from
shared encoder, as input to SS and actor heads
"""


def evaluate_agent(agent, env, args, buffer=None, step=None, feat_analysis=False, L=None): # OK
    """Evaluate agent on env, storing obses, actions and next obses
    Params : - agent : IL agent
             - env : env to evaluate this agent in
             - args
             - buffer : needed to train IL agent with a trajectory as input to the domain specific module"""

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
            # Trajectory : (obs, act, obs, act, obs)
            traj = None if buffer is None else buffer.sample_traj()
            feat_vects.append(agent.extract_feat_vect(traj))
            with utils.eval_mode(agent):
                action = agent.select_action(obs, traj)
            next_obs, reward, done, info, _, _ = env.step(action, rewards)
            episode_reward += reward
            if L and step:
                L.log('eval/episode_reward', episode_reward, step)
            obses.append(obs)
            actions.append(action)
            obs = next_obs
            step += 1

        obses.append(obs)
        ep_rewards.append(episode_reward)

    if L and step:
        L.dump(step)

    if feat_analysis :
        return np.array(ep_rewards), obses, actions, feat_vects
    return np.array(ep_rewards), obses, actions

def collect_trajectory(RL_reference, env, args):
    """Collect trajectory on domain with a reference RL agent
    which has not been trained on the given domain"""

    buffer = utils.TrajectoryBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=10000,
        batch_size=args.batch_size
    )
    _, obses, actions = evaluate_agent(RL_reference, env, args)
    buffer.add_path(obses, actions)

    return buffer


def collect_expert_samples(agent, env, args, label): # OK
    """Collect data samples for Imitation learning training
       Args : - agent : expert RL agent trained on env
              - env : dm_control environment
              - args
              - label : env specificity, eg. the cartmass
       """
    # Create replay buffer with label
    buffer = utils.ExtendedTrajectoryBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.train_steps,
        batch_size=args.batch_size,
        label=label
    )

    ep_rewards, obses, actions = evaluate_agent(agent, env, args)
    buffer.add_path(obses, actions) # add policy rollout to buffer
    return buffer, ep_rewards.mean(), ep_rewards.std()

def relabel(obses, expert): # OK
    """Relabel observations with expert agent for DAgger algorithm"""
    with utils.eval_mode(expert):
        actions_new = []
        for obs in obses:
            actions_new.append(expert.select_action(obs))
    return actions_new

def load_agent(label, action_shape, args): # OK
    """Load RL agent model from directory"""

    work_dir = args.work_dir + label # example : logs/cartpole_swingup + "_0_3"
    L = Logger(args.save_dir + label, use_tb=True, config='il')
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


    # TODO : change labels for force folders
    labels = ["_0_4", "_0_2", "_0_25", "_0_3"]

    # 1. Define 4 envts : four different environments / domains that differ by the mass or the force
    print("-"*60)
    print("Define environment")
    envs = []
    # TODO replace with forces
    masses = [] # eg. for cartpole [1, 0.1] : mass of the cart and mass of the pole
    for mass in [0.4, 0.2, 0.25, 0.3]: # TODO replace with forces values
        env = init_env(args, mass)
        masses.append(env.get_masses()) # TODO env.get_forces()
        envs.append(env)

    # 2. Load expert agents + reference agent
    print("-" * 60)
    print("Load experts")
    experts = []
    loggers = []
    for label in labels:
        # All envs have should have the same action space shape
        agent, logger = load_agent(label, envs[0].action_space.shape, args)
        experts.append(agent)
        loggers.append(logger)
    # Load reference agent
    # ref_expert, _ = load_agent("", envs[0].action_space.shape, args)


    # 3. Collect samples from 4 RL agents
    print("-" * 60)
    print("Fill in buffers")
    buffers = [] # save data for IL
    stats_expert = dict() # save score of trained RL agents on corresponding environments
    stats_il = {k:[] for k in labels} # save score of Il agents

    # 4. Initialize buffers by collecting experts data and collect their performance in the meantime

    # 4.a We have 1 buffer per (env, RL_expert)
    for expert, mass, env in zip(experts, labels, envs) : # TODO replace mass with force labels
        buffer, mean, std = collect_expert_samples(expert, env, args, mass)
        buffers.append(buffer)
        stats_expert[mass] = [mean, std]

    # 4.b Collect trajectories from ref RL agent on different domains
    # trajs_buffers = []
    # for env in envs:
    #     trajs_buffers.append(collect_trajectory(ref_expert, env, args))

    # 5. Create IL agents
    print("-" * 60)
    print("Create IL agents")
    il_agents = []
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)

    # TODO replace with force values
    for mass in masses:
        il_agent = make_il_agent_visual(
            obs_shape=cropped_obs_shape,
            action_shape=envs[0].action_space.shape,
            args=args)
        il_agents.append(il_agent)

    # Share domain generic part between agents
    il_agents_tied = [il_agents[0]]
    for il_agent in il_agents[1:]:
        il_agent.tie_agent_from(il_agents[0])
        il_agents_tied.append(il_agent)

    # 6. Train the four IL agents with DAgger algorithm
    print("-" * 60)
    print("Train IL agents")

    for it in range(args.n_iter): # number of dagger iterations
        print("\n\n********** Training %i ************"%it)

        # a. Train 4 Il agents policies
        for step in range(args.il_steps):

            # Store action predictions, gt, losses
            preds, pred_invs, gts, losses = [], [], [], 0

            # Forward pass sequentially for all agents
            #for agent, buffer, traj_buffer, mass, L in zip(il_agents_tied, buffers, trajs_buffers, masses, loggers): # TODO replace with forces
            for agent, buffer, mass, L in zip(il_agents_tied, buffers, masses,
                                                               loggers):

                # sample a batch of obs, action, next_obs and traj = [obs1, act1, obs2, act2, obs3]
                obs, action, next_obs, traj = buffer.sample() # change again without traj
                #traj = traj_buffer.sample()
                action_pred, action_inv, loss = agent.predict_action(obs, next_obs, traj, action, L=L, step=step)

                preds.append(action_pred) # Action from actor network
                pred_invs.append(action_inv) # Action from SS head
                gts.append(action) # Ground truth action (from expert)
                losses += loss

            # Backward pass
            losses.backward()

            for agent in il_agents_tied:
                agent.update()

        # b. Evaluate - Perform IL agent policy rollouts
        print("\n\n********** Evaluation and relabeling %i ************" % it)
        # TODO replace mass labels with force labels
        #for agent, expert, logger, env, buffer, traj_buffer, mass in zip(il_agents_tied, experts, loggers, envs, buffers, trajs_buffers, labels):
        for agent, expert, logger, env, buffer, mass in zip(il_agents_tied, experts, loggers, envs,
                                                                             buffers, labels):

            # Evaluate agent on envt
            rewards, obses, actions = evaluate_agent(agent, env, args, buffer, L=logger, step=it) # change to traj_buffer
            # Save itermediary score
            stats_il[mass].append([rewards.mean(), rewards.std()])
            print(f'Performance of agent on mass {mass} : {rewards.mean()} +/- {rewards.std()}')
            # Relabel actions -> using DAgger algorithm
            actions_new = relabel(obses, expert)
            # Add trajectory to training buffer
            buffer.add_path(obses, actions_new)


        # c. Save partial model
        if it % 5 == 0 :
            for agent, label in zip(il_agents_tied, labels):
                save_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
                agent.save(save_dir, it)


    # 7. Save IL agents
    for agent, label in zip(il_agents_tied, labels):
        save_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
        agent.save(save_dir, "final")

    # 8. Evaluate expert vs IL
    for label in labels:
        print("-" * 60)
        print(f'Mass of {label}')
        print(f'Expert performance : {stats_expert[label][0]} +/- {stats_expert[label][1]}')
        print(f'Imitation learning agent with dagger performance : {stats_il[label][-1][0]} +/- {stats_il[label][-1][1]}')

def test_agents(args):

    # Load env for a given mass
    envs = []
    masses = []
    for label in [0.3, 0.2, 0.25, 0.4]:
        env = init_env(args, label)
        masses.append(env.get_masses())
        envs.append(env)

    # Build traj buffers
    traj_buffers = []
    ref_expert, _ = load_agent("", envs[0].action_space.shape, args)
    for env in envs:
        traj_buffers.append(collect_trajectory(ref_expert, env, args))

    for label, mass, traj_buffer, env in zip(["_0_3", "_0_2", "_0_25", "_0_4"], masses, traj_buffers, envs):
        # Load IL agent
        cropped_obs_shape = (3 * args.frame_stack, 84, 84)
        il_agent = make_il_agent_visual(
            obs_shape=cropped_obs_shape,
            action_shape=envs[0].action_space.shape,
            args=args)
        load_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
        il_agent.load(load_dir, "final")
        # Evaluate agent
        reward, _, _ = evaluate_agent(il_agent, env, args, buffer=traj_buffer)
        print('non adapting reward:', int(reward.mean()), ' +/- ', int(reward.std()), ' for label ', label)

if __name__ == '__main__':
    args = parse_args()
    test_agents(args)