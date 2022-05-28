import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import utils
from video import VideoRecorder
from arguments import parse_args
from agent.IL_agent_visual import make_il_agent_visual
from eval import init_env, evaluate
from il.imitation_learning_visual import evaluate_agent, collect_trajectory, load_agent
from agent.IL_agent import make_il_agent

def setup(args, domains, labels):

    """Load IL agents and corresponding envs for testing"""

    # TODO generalize to non visual and use it in the below functions
    # TODO generalize to forces

    envs = []
    masses = []
    for label in domains:
        env = init_env(args, label)
        masses.append(env.get_masses())
        envs.append(env)

    il_agents = []
    for label, mass in zip(labels, masses):
        # Load IL agent
        cropped_obs_shape = (3 * args.frame_stack, 84, 84)
        il_agent = make_il_agent_visual( # TODO modify for non visual based
            obs_shape=cropped_obs_shape,
            action_shape=envs[0].action_space.shape,
            args=args)
        load_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
        il_agent.load(load_dir, "final")
        il_agents.append(il_agent)

    return envs, masses, il_agents


def verify_weights(args):
    """Verify if agents indeed share weights"""

    envs, masses, il_agents = setup(args, [0.3, 0.2], ["_0_3", "_0_2"])

    for agt in il_agents:
        print(agt.actor.encoder.fc.state_dict())
        print(agt.ss_encoder.fc.state_dict())

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

    for _, vect in pca_decomposition.items():
        plt.scatter(vect[:,0], vect[:,1])

    plt.legend([domain for domain in pca_decomposition.keys()])
    plt.grid()
    plt.savefig("images/pca_decomp.jpeg")


def feature_vector_analysis(args):

    print("load agents")
    # Load envs and agents
    envs, masses, il_agents = setup(args, [0.3, 0.2, 0.25, 0.4], ["_0_3", "_0_2", "_0_25", "_0_4"] )

    print("load traj buffers")
    # Build traj buffers
    traj_buffers = []
    ref_expert, _ = load_agent("", envs[0].action_space.shape, args)
    for env in envs:
        traj_buffers.append(collect_trajectory(ref_expert, env, args))

    print("extract features")
    # Extract feat vects from Il agents
    features = dict()
    for label, env, buffer, il_agent in zip(["_0_3", "_0_2", "_0_25", "_0_4"], envs, traj_buffers, il_agents):
        _, _, _, feat_vects = evaluate_agent(il_agent, env, args, feat_analysis=True, buffer=buffer)
        features[label] = np.array(feat_vects)

    print("perform PCA")
    # Perform PCA analysis
    PCA_decomposition(features)

def main(args):

    """Try test time adaption of IL agents"""

    domain = [args.domain_test]
    label = [args.label]
    rd = args.rd
    print(f'domain {domain} label {label} at random' if rd else f'domain {domain} label {label}')

    # 1. Load agent

    # Load environment
    envs, masses, il_agents = setup(args, domain, label)
    il_agent, env = il_agents[0], envs[0]

    # Initialize feature vector either at random either with domain_specific feature vector
    # if rd :
    #     init = np.random.rand(args.dynamics_output_shape)
    # else :
    #     init = il_agent.extract_feat_vect([0.3, 0.1])
    # il_agent.init_feat_vect(init)

    # 2. Prepare test time evaluation
    # Build traj buffers
    ref_expert, _ = load_agent("", env.action_space.shape, args)
    traj_buffer = collect_trajectory(ref_expert, env, args)

    # 3. Non adapting agent
    reward, _, _ = evaluate_agent(env, il_agent, args, buffer=traj_buffer, adapt=False)
    print('non adapting reward:', int(reward.mean()), ' +/- ', int(reward.std()))

    # 4 . Adapting agent
    env = init_env(args, domain)
    print(f'Policy Adaptation during Deployment for IL agent of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
    pad_reward, _, _ = evaluate_agent(env, il_agent, args, buffer=traj_buffer, adapt=True)
    print('pad reward:', int(pad_reward.mean()), ' +/- ', int(pad_reward.std()))


if __name__ == "__main__":
    args = parse_args()
    main(args)
    