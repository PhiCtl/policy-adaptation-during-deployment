import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from arguments import parse_args
from utils_imitation_learning import evaluate_agent, eval_adapt, setup, setup_small

"""Script to :
    - analyse the feature vectors of the domain specific module
    - perform test time training of IL agents
    - verify if weights were correctly tied
    - evaluate agents again on their training environment"""


def verify_weights(args):
    """Verify if agents indeed share weights"""

    envs, masses, il_agents = setup_small(args, [0.4, 0.2, 0.25, 0.3], ["_0_4", "_0_2", "_0_25", "_0_3"])

    print(il_agents[0].verify_weights_from(il_agents[1]))
    print("-"*60)
    print(il_agents[1].verify_weights_from(il_agents[2]))
    print("-" * 60)
    print(il_agents[2].verify_weights_from(il_agents[3]))
    #
    a0, a1 = il_agents[0], il_agents[1]
    print(a1.actor.encoder.convs[0].weight, a0.actor.encoder.convs[0].weight)
    print("-" * 60)
    print(a1.actor.encoder.convs[1].bias, a0.actor.encoder.convs[1].bias)

def PCA_decomposition(groups):


    """Perform PCA decomposition into 2 principal components
    of each group in groups and plot the result (in 2D)"""

    pca_decomposition = dict()

    # Perform PCA decomposition
    for domain, vect in groups.items():
        std_data = StandardScaler().fit_transform(groups[domain])
        pca = PCA(n_components=2)
        pca_decomposition[domain] = pca.fit_transform(std_data)

    # Plot
    plt.figure(figsize=(20,20))
    plt.xlabel('Principal Component 1', fontsize=15)
    plt.ylabel('Principal Component 2', fontsize=15)
    plt.title('2 component PCA', fontsize=20)

    for domain, vect in pca_decomposition.items():
        plt.scatter(vect[:,0], vect[:,1], label=domain)

    plt.legend()
    plt.grid()
    plt.savefig("images/pca_decomp.jpeg")


def feature_vector_analysis(args):

    print("load agents")
    # Load envs and agents
    envs, masses, il_agents = setup(args, [0.3, 0.2, 0.25, 0.4], ["_0_3", "_0_2", "_0_25", "_0_4"] )

    # print("load traj buffers")
    # Build traj buffers
    traj_buffers = []
    # ref_expert, _ = load_agent("", envs[0].action_space.shape, args)
    # for env in envs:
    #     traj_buffers.append(collect_trajectory(ref_expert, env, args))

    print("extract features")
    # Extract feat vects from Il agents
    features = dict()
    for label, env, il_agent in zip(["_0_3", "_0_2", "_0_25", "_0_4"], envs, il_agents):
        _, _, _, feat_vects = evaluate_agent(il_agent, env, args, feat_analysis=True, buffer=None)
        features[label[1:]] = np.array(feat_vects)

    print("perform PCA")
    # Perform PCA analysis
    PCA_decomposition(features)

def seed_screening(args, num_seeds=5):

    """For GT Il agents only"""
    adapt_rw, rw = [], []

    for i in range(num_seeds):

        # Load environment
        envs, masses, il_agents = setup_small(args, [args.domain], [args.label], seed=i)
        il_agent, env = il_agents[0], envs[0]

        # Initialize feature vector either at random either with domain_specific feature vector
        if args.rd:
            init = np.random.rand((2, 1))
        else:
            init = il_agent.extract_feat_vect([args.domain_training, 0.1])  # [tgt_domain, 0.1]
        il_agent.init_feat_vect(init, batch_size=args.pad_batch_size)

        # Non adapting agent
        reward, _, _ = eval_adapt(il_agent, env, args)
        rw.append(reward)
        print('non adapting reward:', int(reward.mean()), ' +/- ', int(reward.std()), ' for label ', args.label)

        # Adapting agent
        print(
            f'Policy Adaptation during Deployment for IL agent of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
        reward, _, _ = eval_adapt(il_agent, env, args, adapt=True)
        adapt_rw.append(reward)
        print('pad reward:', int(reward.mean()), ' +/- ', int(reward.std()), ' for label ', args.label)

    adapt_rw = np.array(adapt_rw)
    print(f'Adapting agent performance : {adapt_rw.mean()} +/- {adapt_rw.std()}')
    rw = np.array(rw)
    print(f'Non adapting agent performance : {rw.mean()} +/- {rw.std()}')


def lr_screening(il_agent, label, env, args, lrs=[1e-4, 1e-3, 1e-2, 1e-1, 0.5]):

    """For GIT IL agents only at the moment"""
    # TODO generalize to other agents

    # Non adapting agent
    reward, _, _ = eval_adapt(il_agent, env, args)
    print('non adapting reward:', int(reward.mean()), ' +/- ', int(reward.std()), ' for label ', label)

    for lr in lrs:
        # Adapting agent
        il_agent.il_lr = lr
        print(
            f'Policy Adaptation during Deployment for IL agent of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
        reward, _, _ = eval_adapt(il_agent, env, args, adapt=True)
        print('pad reward:', int(reward.mean()), ' +/- ', int(reward.std()), ' for label ', label, ' and lr ', lr)



def main(args):

    """Try test time adaption of IL agents"""

    print(f'domain {args.domain_test} label {args.label} at random' if args.rd
          else f'domain {args.domain} label {args.label} initialized on {args.domain_training}')
    print(f'learning rate {args.il_lr}')

    envs, masses, il_agents = setup_small(args, [args.domain], [args.label])
    lr_screening(il_agents[0], args.label, envs[0], args, lrs=[0.005, 0.1, 0.5])



def test_agents(args):

    # envs, masses, il_agents_train = setup(args,
    #                                 [0.4, 0.3, 0.25, 0.2],
    #                                 ["_0_4", "_0_3", "_0_25", "_0_2"])

    # for agent, env, label in zip(il_agents_train, envs, ["_0_4", "_0_3", "_0_25", "_0_2"]):
    #     rewards, _, _, _ = evaluate_agent(agent, env, args, dyn=True, buffer=None)
    #     print(f'For {label} agent : {rewards.mean()} +/- {rewards.std()}')

    il_agents, experts, envs, dynamics, buffers, trajs_buffers, stats_expert = setup(args,
                                                                                           train_IL=False,
                                                                                           checkpoint="final",
                                                                                           gt=False)

    # Verify weights
    print(il_agents[0].verify_weights_from(il_agents[1]))
    print("-" * 60)
    print(il_agents[1].verify_weights_from(il_agents[2]))
    print("-" * 60)
    print(il_agents[2].verify_weights_from(il_agents[3]))

    for agent, env, traj, label in zip(il_agents, envs, trajs_buffers, ["_0_4", "_0_2", "_0_25", "_0_3"]):

        rewards, _, _, _ = evaluate_agent(agent, env, args, buffer=traj)
        print(f'For {label} agent : {rewards.mean()} +/- {rewards.std()}')




if __name__ == "__main__":
    args = parse_args()
    main(args)