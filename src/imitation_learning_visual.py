from tqdm import tqdm
from arguments import parse_args
from utils_imitation_learning import *

"""
This script trains Imitation learning agents from RL experts trained on different domains (different by their dynamics
parameters, either the mass or the force).
Those IL agents have a shared domain generic module, which has the same architecture as the original "PAD" agent we've
explored so far. They have their own domain specific module, which takes as input the past 3 observations and actions 
(ie. obs, act, obs, act, obs) to infer the dynamics. The output of this module is concatenated with output from
shared encoder, as input to SS and actor heads
"""


def main(args):

    labels = ["_0_4", "_0_2"]#, "_0_25", "_0_3"],
    domains = [0.4, 0.2 ] #, 0.25, 0.3]
    stats_il = {k: [] for k in labels}  # save score of Il agents

    il_agents, experts, envs, _, buffers, trajs_buffers, stats_expert = setup(args,
                                                                                 labels=labels,
                                                                                 domains=domains)
    # Share domain generic part between agents
    il_agents_train = [il_agents[0]]
    for il_agent in il_agents[1:]:
        il_agent.tie_agent_from(il_agents_train[0])
        il_agents_train.append(il_agent)

    # 6. Train the four IL agents with DAgger algorithm
    print("-" * 60)
    print("Train IL agents")

    for it in tqdm(range(args.n_iter)): # number of dagger iterations
        print("\n\n********** Training %i ************"%it)

        # a. Train 4 Il agents policies
        for step in tqdm(range(args.il_steps)):

            # Store action predictions, gt, losses
            preds, pred_invs, gts, losses = [], [], [], 0

            # Forward pass sequentially for all agents
            for agent, buffer, traj_buffer in zip(il_agents_train, buffers, trajs_buffers):

                # sample a batch of obs, action, next_obs and traj = [obs1, act1, obs2, act2, obs3]
                obs, action, next_obs = buffer.sample()
                traj = traj_buffer.sample()
                action_pred, action_inv, loss = agent.predict_action(obs, next_obs, traj, action)

                preds.append(action_pred) # Action from actor network
                pred_invs.append(action_inv) # Action from SS head
                gts.append(action) # Ground truth action (from expert)
                losses += loss

            # Backward pass
            if step % 1000 == 0 : print(losses)
            losses.backward()

            for agent in il_agents_train:
                agent.update()

        # b. Evaluate - Perform IL agent policy rollouts
        print("\n\n********** Evaluation and relabeling %i ************" % it)

        for agent, expert, env, buffer, traj_buffer, mass in zip(il_agents_train, experts, envs, buffers, trajs_buffers, labels):

            # Evaluate agent on envt
            rewards, obses, actions, _ = evaluate_agent(agent, env, args, buffer=traj_buffer)
            # Save intermediary score
            stats_il[mass].append([rewards.mean(), rewards.std()])
            print(f'Performance of agent on mass {mass} : {rewards.mean()} +/- {rewards.std()}')
            # Relabel actions -> using DAgger algorithm
            actions_new = relabel(obses, expert)
            # Add trajectory to training buffer
            buffer.add_path(obses, actions_new)


        # c. Save partial model
        if it % 2 == 0 :
            for agent, label in zip(il_agents_train, labels):
                save_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
                agent.save(save_dir, it)


    # 7. Save IL agents
    for agent, label in zip(il_agents_train, labels):
        save_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
        agent.save(save_dir, "final")

    # Final evaluation
    for agent, env, traj_buffer, mass in zip(il_agents_train, envs, trajs_buffers,
                                                             labels):
        # Evaluate agent on envt
        rewards, obses, actions, _ = evaluate_agent(agent, env, args, buffer=traj_buffer)
        # Save intermediary score
        stats_il[mass].append([rewards.mean(), rewards.std()])
        print(f'Performance of agent on mass {mass} : {rewards.mean()} +/- {rewards.std()}')

    # 8. Evaluate expert vs IL
    for label in labels:
        print("-" * 60)
        print(f'Mass of {label}')
        print(f'Expert performance : {stats_expert[label][0]} +/- {stats_expert[label][1]}')
        print(f'Imitation learning agent with dagger performance : {stats_il[label][-1][0]} +/- {stats_il[label][-1][1]}')


if __name__ == '__main__':
    args = parse_args()
    main(args)