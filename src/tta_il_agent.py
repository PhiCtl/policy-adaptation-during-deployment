import numpy as np

from arguments import parse_args
from recorder import AdaptRecorder
from utils_imitation_learning import evaluate_agent, eval_adapt, setup, setup_small

"""Script to :
    - perform test time training of IL agents
    - verify if weights were correctly tied
    - evaluate agents again on their training environment"""


def verify_weights(args, domains, labels, mass, visual):
    """Verify if agents indeed share weights
    PArams : - args
             - domains : [0.4, 0.3] for instance for different cart masses
             - labels : corresponding labels ["_0_4", "_0_3"] for instance
             - mass : boolean either mass dynamics change either force (in this case set mass to False)
             - visual : boolean for visual input based agents """

    envs, masses, il_agents = setup_small(args, domains, labels, mass=mass, visual=visual)

    print(il_agents[0].verify_weights_from(il_agents[1]))
    print("-"*60)
    print(il_agents[1].verify_weights_from(il_agents[2]))
    print("-" * 60)
    print(il_agents[2].verify_weights_from(il_agents[3]))


def seeds_summary(args, num_seeds=6, lr=None):

    """For GT Il agents only
    Params : - args
             - num_seeds : on which we can evaluate our agents
             - lr : the learning rate for test time adaptation"""

    adapt_rw, rw = [], []
    adapt_recorder = AdaptRecorder(args.work_dir, args.mode)
    recorder = AdaptRecorder(args.work_dir, args.mode)

    for i in range(num_seeds):

        # Load environment
        envs, forces, il_agents = setup_small(args, [args.domain_test], [args.label], seed=i, visual=False, mass=True)
        il_agent, env = il_agents[0], envs[0]
        if lr: il_agent.il_lr = lr

        # Initialize feature vector either at random either with domain_specific feature vector
        if args.rd:
            init = np.ones(args.dynamics_output_shape)*1000
        else:
            init = il_agent.extract_feat_vect([args.domain_training, 0.1])  # TODO change for forces
        il_agent.init_feat_vect(init, batch_size=args.pad_batch_size)

        # Non adapting agent
        reward, _, _ = eval_adapt(il_agent, env, args, recorder=recorder)
        rw.append(reward)
        print('non adapting reward:', int(reward.mean()), ' +/- ', int(reward.std()), ' for label ', args.label)

        # Adapting agent
        print(
            f'Policy Adaptation during Deployment for IL agent of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
        reward, _, _ = eval_adapt(il_agent, env, args, adapt=True, recorder=adapt_recorder)
        adapt_rw.append(reward)
        print('pad reward:', int(reward.mean()), ' +/- ', int(reward.std()), ' for label ', args.label)

    adapt_rw = np.array(adapt_rw)
    print(f'Adapting agent performance : {adapt_rw.mean()} +/- {adapt_rw.std()}')
    dom = "_rd" if args.rd else "_" + str(args.domain_training)
    adapt_recorder.save("performance_" + str(lr) + dom, adapt=True)
    rw = np.array(rw)
    print(f'Non adapting agent performance : {rw.mean()} +/- {rw.std()}')
    recorder.save("performance_" + str(lr) + dom, adapt=False)

def seeds_summary_visual(args, lr=None, num_seeds=6):

    """Same function as above but for visual input based imitation learning agents"""

    adapt_rw, rw = [], []
    adapt_recorder = AdaptRecorder(args.work_dir, args.mode)
    recorder = AdaptRecorder(args.work_dir, args.mode)

    for i in range(num_seeds):

        # Load environment
        [il_agent], _, [env], _, _, [traj_buffer], _ = setup(args,
                                                             labels=[args.label],
                                                             domains=[args.domain_test],
                                                             train_IL=False,
                                                             checkpoint="final",
                                                             gt=False,
                                                             seed=i)
        # Change agent adaptation learning rate
        if lr: il_agent.il_lr = lr

        # Non adapting agent
        reward, _, _ = evaluate_agent(il_agent, env, args, buffer=traj_buffer, recorder=recorder)
        rw.append(reward)
        print('non adapting reward:', int(reward.mean()), ' +/- ', int(reward.std()), ' for label ', args.label)

        # Adapting agent
        print(
            f'Policy Adaptation during Deployment for IL agent of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
        reward, _, _ = evaluate_agent(il_agent, env, args, buffer=traj_buffer, recorder=adapt_recorder, adapt=True)
        adapt_rw.append(reward)
        print('pad reward:', int(reward.mean()), ' +/- ', int(reward.std()), ' for label ', args.label)

    adapt_rw = np.array(adapt_rw)
    print(f'Adapting agent performance : {adapt_rw.mean()} +/- {adapt_rw.std()}')
    dom = "_rd" if args.rd else "_" + str(args.domain_training)
    adapt_recorder.save("performance_visual_" + str(lr) + dom, adapt=True)
    rw = np.array(rw)
    print(f'Non adapting agent performance : {rw.mean()} +/- {rw.std()}')
    recorder.save("performance_visual_" + str(lr) + dom, adapt=False)


def main(args):

    """Try test time adaption of IL agents"""

    print(f'domain {args.domain_test} label {args.label} at random' if args.rd
          else f'domain {args.domain_test} label {args.label} initialized on {args.domain_training}')
    print(f'learning rate {args.il_lr}')

    for lr in [0.00001, 0.00005] : #[0.0001,0.001, 0.005, 0.01, 0.05, 0.1]:
        print("Learning rate :", lr)
        seeds_summary_visual(args, lr=lr, num_seeds=3) # change to seeds_summary_visual(args, lr=lr) if needed

def test_agents(args):


    # Uncomment part below to test ground truth input based agents
    envs, masses, il_agents_train = setup_small(args,
                                    [0.4, 0.3, 0.25, 0.2],
                                    ["_0_4", "_0_3", "_0_25", "_0_2"])


    for agent, label, mass in zip(il_agents_train, ["_0_4", "_0_3", "_0_25", "_0_2"], masses):
        print(f'Label {label}')
        print("-"*60)
        for env, env_lab in zip(envs, ["_0_4", "_0_3", "_0_25", "_0_2"]):
            init = np.ones(args.dynamics_output_shape)*10
            agent.init_feat_vect(init, batch_size=args.pad_batch_size)
            rewards, _, _ = eval_adapt(agent, env, args)
            print(f'For {label} agent and env {env_lab} : {rewards.mean()} +/- {rewards.std()}')

    # Test of visual input based agents
    # labels = ["_0_4", "_0_3", "_0_2", "_0_25"] # Shoudl be in same order as domaines
    # il_agents, experts, envs, dynamics, buffers, trajs_buffers, stats_expert = setup(args,
    #                                                                                  domains=[0.4, 0.3, 0.2, 0.25],
    #                                                                                  labels = labels,
    #                                                                                  train_IL=False,
    #                                                                                  checkpoint="final",
    #                                                                                  gt=False)
    #
    # # Verify weights
    # print(il_agents[0].verify_weights_from(il_agents[1]))
    # print("-" * 60)
    # print(il_agents[1].verify_weights_from(il_agents[2]))
    # print("-" * 60)
    # print(il_agents[2].verify_weights_from(il_agents[3]))
    #
    # for agent, env, traj, label in zip(il_agents, envs, trajs_buffers, labels):
    #
    #     rewards, _, _ = evaluate_agent(agent, env, args, buffer=traj)
    #     print(f'For {label} agent : {rewards.mean()} +/- {rewards.std()}')


if __name__ == "__main__":
    args = parse_args()
    #verify_weights(args, [0.4, 0.3, 0.25, 0.2],["_0_4", "_0_3", "_0_25", "_0_2"], mass=True, visual=False)
    main(args)