
from tqdm import tqdm

from arguments import parse_args
from utils_imitation_learning import *
"""
This script trains Imitation learning agents from RL experts trained on different domains (different by their dynamics
parameters, either the mass or the force).
Those IL agents have a shared domain generic module, which has the same architecture as the original "PAD" agent we've
explored so far. They have their own domain specific module, which takes as input the groundtruth (ie. actual)
dynamics value, eg. the mass of the system. The output of this module is concatenated with the output from
shared encoder, as input to SS and actor heads
"""

    
def main(args):

    labels = ["_0_4", "_0_2", "_0_25", "_0_3"] #for cart-pole mass
    #labels = ["_0_-1", "_0_-2", "_0_-3"] #for walker-walk force
    domains = [0.4, 0.2, 0.25, 0.3] #for cart-pole mass
    #domains = [-1, -2, -3] #for walker-walk force
    # TODO change below for forces
    il_agents, experts, envs, dynamics, buffers, _, stats_expert = setup(args,
                                                                         labels=labels,
                                                                         domains=domains,
                                                                         gt=True,
                                                                         train_IL=True,
                                                                         type="mass")
    il_agents_train = [il_agents[0]]
    for il_agent in il_agents[1:]:
        il_agent.tie_agent_from(il_agents_train[0])
        il_agents_train.append(il_agent)

    # print("Verify weights")
    for i in range(len(il_agents_train) - 1):
        print(il_agents_train[i].verify_weights_from(il_agents_train[i + 1]))

    stats_il = {k:[] for k in labels} # save score of Il agents

    # Train the domains IL agents with DAgger algorithm
    print("-" * 60)
    print("Train IL agents")

    for it in tqdm(range(args.n_iter)): # number of dagger iterations
        print("\n\n********** Training %i ************"%it)

        # Train domains Il agents policies
        for step in tqdm(range(args.il_steps)):

            # Save data
            preds, pred_invs, gts, losses = [], [], [], 0

            # Forward pass sequentially for all agents
            for agent, buffer, mass in zip(il_agents_train, buffers, dynamics):
                obs, action, next_obs = buffer.sample() # sample a batch
                action_pred, action_inv, loss = agent.predict_action(obs, next_obs, action, mass=mass) # TODO change here for forces

                preds.append(action_pred) # Action from actor network
                pred_invs.append(action_inv) # Action from SS head
                gts.append(action)
                losses += loss

            # Backward pass
            losses.backward()
            if step % 1000 == 0: print(losses)

            for agent in il_agents_train:
                agent.update()

        # Evaluate - Perform IL agent policy rollouts
        print("\n\n********** Evaluation and relabeling %i ************" % it)
        for agent, expert, env, buffer, mass in zip(il_agents_train, experts, envs, buffers, labels):
            # evaluate agent on environment
            # TODO change for forces
            rewards, obses, actions = eval_adapt(agent, env, args, mass=True, adapt=False, train=True)
            stats_il[mass].append([rewards.mean(), rewards.std()]) # save intermediary score
            print(f'Performance of agent on mass {mass} : {rewards.mean()} +/- {rewards.std()}')
            actions_new = relabel(obses, expert)
            buffer.add_path(obses, actions_new)


        # Save partial model
        if it % 2 == 0 :
            for agent, label in zip(il_agents_train, labels):
                save_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
                agent.save(save_dir, it)


    # 6. Save IL agents
    for agent, label in zip(il_agents_train, labels):
        save_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
        agent.save(save_dir, "final")

    # print("Verify weights")
    for i in range(len(il_agents_train) - 1):
        print(il_agents_train[i].verify_weights_from(il_agents_train[i + 1]))

    # 7. Evaluate expert vs IL
    for label in labels :
        print("-"*60)
        print(f'Mass of {label}')
        print(f'Expert performance : {stats_expert[label][0]} +/- {stats_expert[label][1]}')
        print(f'Imitation learning agent with dagger performance : {stats_il[label][-1][0]} +/- {stats_il[label][-1][1]}')


if __name__ == '__main__':
    args = parse_args()
    main(args)