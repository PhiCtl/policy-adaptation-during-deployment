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
from agent.agent_domain_generic import make_domain_generic_agent
import utils
from eval import init_env, evaluate
from logger import Logger

def evaluate_agent(agent, env, args, buffer=None, step=None, L=None): # OK
    """Evaluate agent on env, storing obses, actions and next obses in buffer if any"""

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
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
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

    return np.array(ep_rewards), obses, actions


def collect_expert_samples(agent, env, args, label = None): # OK
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
    buffer.add_path(obses, actions)
    return buffer, ep_rewards.mean(), ep_rewards.std()

def relabel(obses, expert): # OK
    """Relabel observations with expert agent"""
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
    #labels = ["_0_4", "_0_2", "_0_25", "_0_3"]
    labels = ["_0_-1", "_0_-2", "_0_-3"]
    
    # Define 4 envts
    print("-"*60)
    print("Define environment")
    #all the enviroments for different domains
    envs = []
    #masses = []
    forces = []
    for force in [-1, -2, -3]:
        env = init_env(args, force)
        forces.append(env.get_forces())
        #print(masses[-1]) # debug
        envs.append(env)

    # Load expert agents
    print("-" * 60)
    print("Load experts")
    experts = []
    #loggers = []
    
    for label in labels:
        # All envs have should have the same action space shape
        agent, _ = load_agent(label, envs[0].action_space.shape, args)
        experts.append(agent)
        #loggers.append(logger)

    # Collect samples from the domain-generic agent
    print("-" * 60)
    print("Fill in buffer")

    stats_expert = dict()
    stats_domain_generic_agent = {k:[] for k in labels}   # save score of domain generic agent
         
    buffer = utils.SimpleBuffer(
        obs_shape=envs[0].observation_space.shape,
        action_shape=envs[0].action_space.shape,
        capacity=args.train_steps,
        batch_size=args.batch_size
    ) 

    for expert, env, label in zip(experts, envs, labels):
        rewards, obses, actions = evaluate_agent(expert, env, args)
        buffer.add_path(obses, actions) 
        stats_expert[label]= [rewards.mean(), rewards.std()] #performance of the expert agent in its domain

    print("-" * 60)
    print("Create domain generic agent")
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)

    domain_generic_agent = make_domain_generic_agent(
                obs_shape=cropped_obs_shape,
                action_shape=envs[0].action_space.shape,
                args=args)

    # Train the IL domain generic agent with DAgger algorithm
    print("-" * 60)
    print("Train domain generic agent")

    for it in range(args.n_iter): # number of dagger iterations
        print("\n\n********** Training %i ************"%it)

        # Train the domain generic agent policy
        for step in range(args.il_steps):

            # Sample data
            preds, pred_invs, gts, losses = [], [], [], 0

            # Forward pass for the domain generic agent for all domains (?)
            
            obs, action, next_obs, _ = buffer.sample() # sample a batch
            action_pred, action_inv, loss = domain_generic_agent.predict_action(obs, next_obs, action)

            preds.append(action_pred) # Action from actor network
            pred_invs.append(action_inv) # Action from SS head
            gts.append(action)
            losses += loss

            # Backward pass
            losses.backward()

            # Update the domain generic agent
            domain_generic_agent.update()

        #Evaluate - Perform domain-generic agent
        print("\n\n********** Evaluation and relabeling %i ************" % it)
        for expert, env, label in zip(experts, envs, labels):
            rewards, obses, actions = evaluate_agent(domain_generic_agent, env, args)
            stats_domain_generic_agent[label].append([rewards.mean(), rewards.std()])
            print(f'Performance of domain generic agent: {rewards.mean()} +/- {rewards.std()}') 
            actions_new = relabel(obses, expert)
            buffer.add_path(obses, actions_new)

        # Save partial model
        if it % 3 == 0 :
            save_dir = utils.make_dir(os.path.join(args.save_dir, "", 'model'))
            domain_generic_agent.save(save_dir, it)

    # Evaluate domain_generic_agent on environments=different domains
    save_dir = utils.make_dir(os.path.join(args.save_dir, "", 'model'))
    domain_generic_agent.save(save_dir, "final")

    for label in labels:
        print("-"*60)
        #print(f'Baseline performance: {pad_stats[label][0]} +/- {pad_stats[label][1]}')
        print(f'Expert performance : {stats_expert[label][0]} +/- {stats_expert[label][1]}')
        print(f'Imitation learning agent with dagger performance : {stats_domain_generic_agent[label][-1][0]} +/- {stats_domain_generic_agent[label][-1][1]}')


if __name__ == '__main__':
    args = parse_args()
    main(args)