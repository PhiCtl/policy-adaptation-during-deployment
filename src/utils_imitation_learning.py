import numpy as np
import os
import torch
from tqdm import tqdm
from copy import deepcopy

import utils
from agent.agent import make_agent
# from agent.IL_agent import make_il_agent
# from agent.IL_agent_visual import make_il_agent_visual
from agent.IL_agents import make_il_agent
from eval import init_env

def evaluate_agent(agent, env, args, buffer=None, exp_type="",
                   video=None, recorder=None, adapt=False):
    """Evaluate agent on env, storing obses, actions and next obses
    Params : - agent: IL agent visual
             - env: env to evaluate this agent in
             - args
             - buffer: needed to train IL agent with a trajectory as input to the domain specific module"""

    ep_rewards = []
    obses, actions = [], []
    if buffer :
        buff = deepcopy(buffer) # Because we don't want to modify buffer batch size outside the function
        buff.batch_size = args.pad_batch_size


    for i in tqdm(range(args.num_rollouts)):

        # If we adapt agent at test time, we need to each time load a new agent
        if adapt:
            ep_agent = deepcopy(agent)
            ep_agent.train()
        else :
            ep_agent = agent

        if video: video.init(enabled=True)

        obs = env.reset()
        done = False
        episode_reward, step, rewards, losses = 0, 0, [], []

        while not done:

            # Take a step
            # Trajectory : (obs, act, obs, act, obs)
            traj = buff.sample_traj() if buffer else None

            with utils.eval_mode(ep_agent):
                action = ep_agent.select_action(obs, traj)
            next_obs, reward, done, info, change, _ = env.step(action, rewards)

            # Save data
            episode_reward += reward
            obses.append(obs)
            actions.append(action)

            # Adapt
            if adapt and buffer:
                # Prepare batch of observations
                batch_obs = utils.batch_from_obs(torch.Tensor(obs).cuda(), batch_size=args.pad_batch_size)
                batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
                batch_action = torch.Tensor(action).cuda().unsqueeze(0).repeat(args.pad_batch_size, 1)
                batch_traj = buff.sample()

                # Update domain specific module
                ep_agent.update_inv(utils.random_crop(batch_obs), utils.random_crop(batch_next_obs),
                                    batch_action, batch_traj)


            if video: video.record(env, losses)
            if recorder: recorder.update(change, reward)

            obs = next_obs
            step += 1

        obses.append(obs)  # Save last next obs
        ep_rewards.append(episode_reward)
        if video: video.save(f'{args.mode}_{exp_type}_{i}.mp4')
        if recorder: recorder.end_episode()

    return np.array(ep_rewards), obses, actions

def eval_adapt(agent, env, args, mass=True, adapt=False, train=False, video=None, recorder=None):
    """Evaluate agent on env, storing obses, actions and next obses
    Params : - agent : IL agent Ground truth (with the groundtruth dynamics as input to domain specific)
             - env : env to evaluate this agent in
             - mass : if the dynamics change is a mass change
             - train : if we're in training phase
             - adapt : if we're in the adaptation phase and we want to adapt at test time
             - args"""

    ep_rewards = []
    obses, actions =  [], []
    assert(not train or not adapt), "Cannot adapt during training "

    for i in tqdm(range(args.num_rollouts)):

        if adapt:
            ep_agent = deepcopy(agent)
            ep_agent.train()
        else :
            ep_agent = agent

        if video: video.init(enabled=True)

        obs = env.reset()
        done = False
        episode_reward, step, rewards, losses = 0, 0, [], []
        while not done:

            # Take a step
            with utils.eval_mode(ep_agent):
                if train :
                    dyn = env.get_masses() if mass else env.get_forces()
                    action = ep_agent.select_action(obs, mass=dyn) if mass else ep_agent.select_action(obs, force=dyn)
                else : # at evaluation phase based on feature vector inference
                    action = ep_agent.select_action(obs)
            next_obs, reward, done, info, change, _ = env.step(action, rewards)

            # Save data
            episode_reward += reward
            obses.append(obs)
            actions.append(action)

            # Adapt
            if adapt:
                # Prepare batch of observations
                batch_obs = utils.batch_from_obs(torch.Tensor(obs).cuda(), batch_size=args.pad_batch_size)
                batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
                batch_action = torch.Tensor(action).cuda().unsqueeze(0).repeat(args.pad_batch_size, 1)

                losses.append(ep_agent.update_inv(utils.random_crop(batch_obs), utils.random_crop(batch_next_obs),
                                                      batch_action))

            if video: video.record(env, losses)
            if recorder: recorder.update(change, reward)

            obs = next_obs
            step += 1

        obses.append(obs)  # Save last next obs
        ep_rewards.append(episode_reward)
        if video: video.save(f'{args.mode}_pad_{i}.mp4' if adapt else f'{args.mode}_eval_{i}.mp4')
        if recorder: recorder.end_episode()

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


def collect_expert_samples(agent, env, args, label):
    """Collect data samples for Imitation learning training
       Args : - agent : expert RL agent trained on env
              - env : dm_control environment
              - args
              - label : env specificity, eg. the cartmass, the applied force
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
    buffer.add_path(obses, actions) # add policy rollout to buffer
    return buffer, ep_rewards.mean(), ep_rewards.std()

def relabel(obses, expert):
    """Relabel observations with expert agent for DAgger algorithm"""
    with utils.eval_mode(expert):
        actions_new = []
        for obs in obses:
            actions_new.append(expert.select_action(obs))
    return actions_new

def load_agent(label, action_shape, args):
    """Load RL expert agent model from directory"""

    work_dir = args.work_dir + label # example : logs/cartpole_swingup + "_0_3"
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
    agent.load(model_dir, args.pad_checkpoint)# TODO generalize

    return agent

def setup(args,
          labels = ["_0_4", "_0_3", "_0_2", "_0_25"],
          domains = [0.4, 0.3, 0.2, 0.25],
          checkpoint="final",
          type="mass",
          gt=False,
          train_IL=True,
          seed=None):
    """Set up function for IL agents training or evaluating
        Params : - args
                 - labels : list of labels to load agents with
                 - domains : list of domains, either mass or forces, to set up domain specific environment
                             should be in the same order as the labels
                 - type : type of dynamics, either mass or force
                 - gt : if we need il agents trained with ground truth dynamics or with visual input only
                 - train_IL : if we are to train IL agents or evaluate them
                 - seed : for initialisation"""

    assert type in ["mass", "force"], "Dynamics not implemented"
    if seed is None : seed = args.seed

    # 1. Define 4 envts: four different environments / domains that differ by the mass or the force
    print("-" * 60)
    print("Define environment")
    envs = []
    dynamics = []  # eg. for cartpole [1, 0.1] : mass of the cart and mass of the pole
    for d in domains:
        env = init_env(args, mass=d, seed=seed) if type == "mass" else init_env(args, force=d, seed=seed)
        full_dyn = env.get_masses() if type == "mass" else env.get_forces()
        dynamics.append(full_dyn)
        envs.append(env)

    # 2. Load expert agents + reference agent
    print("-" * 60)
    print("Load experts")
    experts = []
    if train_IL :
        for label in labels:
            # All envs have should have the same action space shape
            agent = load_agent(label, envs[0].action_space.shape, args)
            experts.append(agent)
    # Load reference agent
    ref_expert = load_agent("", envs[0].action_space.shape, args)

    # 3. Collect samples from 4 RL agents
    print("-" * 60)
    print("Fill in buffers")
    buffers = []  # save data for IL
    stats_expert = dict()  # save score of trained RL agents on corresponding environments


    # 4. Initialize buffers by collecting experts data and collect their performance in the meantime

    # 4.a We have 1 buffer per (env, RL_expert)
    if train_IL :
        for expert, label, env in zip(experts, labels, envs):
            buffer, mean, std = collect_expert_samples(expert, env, args, label)
            buffers.append(buffer)
            stats_expert[label] = [mean, std]

    # 4.b Collect trajectories from ref RL agent on different domains
    trajs_buffers = []
    for env in envs:
        trajs_buffers.append(collect_trajectory(ref_expert, env, args))

    # 5. Create IL agents
    print("-" * 60)
    print("Create IL agents")
    il_agents = []
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)

    for d, label in zip(dynamics, labels):

        # Either we need agents with ground truth dynamics input
        # if gt : # TODO refractor in a single function
        #     il_agent = make_il_agent(
        #         obs_shape=cropped_obs_shape,
        #         action_shape=envs[0].action_space.shape,
        #         dynamics_input_shape=d.shape[0],
        #         args=args
        #     )
        # # Either we need agents with visual input only
        # else :
        #     il_agent = make_il_agent_visual(
        #         obs_shape=cropped_obs_shape,
        #         action_shape=envs[0].action_space.shape,
        #         args=args)
        il_agent = make_il_agent(obs_shape=cropped_obs_shape,
                                 visual=not gt,
                                action_shape=envs[0].action_space.shape,
                                dynamics_input_shape=d.shape[0],
                                args=args)

        # If test time, we load pre-trained agents
        if not train_IL:
            load_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
            # Two different load functions -> TODO to gather
            if gt : il_agent.load(load_dir, checkpoint)
            else : il_agent = il_agent.load_full(load_dir, checkpoint)

        il_agents.append(il_agent)

    return il_agents, experts, envs, dynamics, buffers, trajs_buffers, stats_expert

def setup_small(args, domains, labels, checkpoint="final", seed=None, visual=False, mass=True):

    """Load IL agents and corresponding envs for testing
    Params : domains : example list for cart mass change/walker walk force change : [0.4, 0.3], [-1, -2, -3]
             labels : corresponding labels to load the files : example ["_0_4", "_0_3"] or ["0_-1", "0_-2", "_0_-3"]
             checkpoint : to load a model
             seed : to initialize env and agent
             visual : If we want a visual or a GT trained IL agent"""
    if seed is None : seed = args.seed

    envs = []
    dyn = []
    for d in domains:
        env = init_env(args, mass=d, seed=seed) if mass else init_env(args, force=d, seed=seed)
        dyn.append(env.get_masses() if mass else env.get_forces())
        envs.append(env)

    il_agents = []
    for label, d in zip(labels, dyn):

        # Load IL agent
        cropped_obs_shape = (3 * args.frame_stack, 84, 84)
        # if visual :
        #     il_agent = make_il_agent_visual(
        #         obs_shape=cropped_obs_shape,
        #         action_shape=envs[0].action_space.shape,
        #         args=args)
        # else :
        #     il_agent = make_il_agent(
        #         obs_shape=cropped_obs_shape,
        #         action_shape=envs[0].action_space.shape,
        #         dynamics_input_shape=d.shape[0],
        #         args=args)
        il_agent = make_il_agent(obs_shape=cropped_obs_shape,
                                 visual=visual,
                                 action_shape=envs[0].action_space.shape,
                                 dynamics_input_shape=d.shape[0],
                                 args=args)
        load_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))

        # I needed two different load functions for IL GT vs IL visual agents
        if visual :
            il_agent = il_agent.load_full(load_dir, checkpoint)
        else :
            il_agent.load(load_dir, checkpoint)

        il_agents.append(il_agent)

    return envs, dyn, il_agents
