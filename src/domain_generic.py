import torch
import os
from tqdm import tqdm
import numpy as np

from arguments import parse_args
from utils_imitation_learning import *
from agent.agent_domain_generic import make_domain_generic_agent
import utils
from eval import init_env

"""Script to train an agent with a domain generic module only, on each of the 3 or 4 considered domains
TODO : change the labels to train on masses"""


def evaluate_agent(agent, env, args):
    """Evaluate agent on env, storing obses, actions and next obses in buffer if any"""

    ep_rewards = []
    obses, actions, feat_vects = [], [], []

    for i in range(args.num_rollouts):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        rewards = []

        while not done:
            # Take a step
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            next_obs, reward, done, info, _, _ = env.step(action, rewards)
            episode_reward += reward

            obses.append(obs)
            actions.append(action)
            obs = next_obs
            step += 1

        obses.append(obs)
        ep_rewards.append(episode_reward)

    return np.array(ep_rewards), obses, actions


def main(args):
    # TODO better practise than lists
    # labels = ["_0_4", "_0_2", "_0_25", "_0_3"]
    labels = ["_0_-1", "_0_-2", "_0_-3"]
    domains = [-1, -2, -3]

    # Define 4 envts
    print("-" * 60)
    print("Define environment")
    # all the enviroments for different domains
    envs = []
    # masses = []
    forces = []
    for force in domains:
        env = init_env(args, force)
        forces.append(env.get_forces())
        # masses.append(env.get_masses())
        envs.append(env)

    # Load expert agents
    print("-" * 60)
    print("Load experts")
    experts = []

    for label in labels:
        # All envs have should have the same action space shape
        agent = load_agent(label, envs[0].action_space.shape, args)
        experts.append(agent)

    # Collect samples for the domain-generic agent
    print("-" * 60)
    print("Fill in buffer")

    stats_expert = dict()
    stats_domain_generic_agent = {k: [] for k in labels}  # save score of domain generic agent

    buffer = utils.SimpleBuffer(
        obs_shape=envs[0].observation_space.shape,
        action_shape=envs[0].action_space.shape,
        capacity=args.train_steps,
        batch_size=args.batch_size
    )

    for expert, env, label in zip(experts, envs, labels):
        rewards, obses, actions = evaluate_agent(expert, env, args)
        buffer.add_path(obses, actions)  # Add all samples in the same buffer
        stats_expert[label] = [rewards.mean(), rewards.std()]  # performance of the expert agent in its domain

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

    for it in tqdm(range(args.n_iter)):  # number of dagger iterations
        print("\n\n********** Training %i ************" % it)

        # Train the domain generic agent policy
        for step in tqdm(range(args.il_steps)):
            # Sample data
            preds, pred_invs, gts, loss = [], [], [], 0

            # Forward pass for the domain generic agent for all domains (?)

            obs, action, next_obs = buffer.sample()  # sample a batch
            action_pred, action_inv, loss = domain_generic_agent.predict_action(obs, next_obs, action)

            preds.append(action_pred)  # Action from actor network
            pred_invs.append(action_inv)  # Action from SS head
            gts.append(action)

            # Backward pass
            loss.backward()

            # Update the domain generic agent
            domain_generic_agent.update()

        # Evaluate - Perform domain-generic agent
        print("\n\n********** Evaluation and relabeling %i ************" % it)
        for expert, env, label in zip(experts, envs, labels):
            rewards, obses, actions = evaluate_agent(domain_generic_agent, env, args )
            stats_domain_generic_agent[label].append([rewards.mean(), rewards.std()])
            print(f'Performance of domain generic agent: {rewards.mean()} +/- {rewards.std()}')
            actions_new = relabel(obses, expert)
            buffer.add_path(obses, actions_new)

        # Save partial model
        if it % 3 == 0:
            save_dir = utils.make_dir(os.path.join(args.save_dir, "", 'model'))
            domain_generic_agent.save(save_dir, it)

    # Evaluate domain_generic_agent on environments=different domains
    save_dir = utils.make_dir(os.path.join(args.save_dir, "", 'model'))
    domain_generic_agent.save(save_dir, "final")

    for label in labels:
        print("-" * 60)
        print(f'Expert performance : {stats_expert[label][0]} +/- {stats_expert[label][1]}')
        print(
            f'Imitation learning agent with dagger performance : {stats_domain_generic_agent[label][-1][0]} +/- {stats_domain_generic_agent[label][-1][1]}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
