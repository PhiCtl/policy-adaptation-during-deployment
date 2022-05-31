import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import utils
from recorder import AdaptRecorder
from video import VideoRecorder
from arguments import parse_args
from agent.IL_agent_visual import make_il_agent_visual
from agent.IL_agent import make_il_agent
from eval import init_env
from utils_imitation_learning import evaluate_agent, collect_trajectory, load_agent, setup

# def setup(args, domains, labels, checkpoint="final"):
#
#     """Load IL agents and corresponding envs for testing"""
#
#     # TODO generalize to non visual and use it in the below functions
#     # TODO generalize to forces
#
#     envs = []
#     masses = []
#     for mass in domains:
#         env = init_env(args, mass)
#         masses.append(env.get_masses())
#         envs.append(env)
#
#     il_agents = []
#     for label, mass in zip(labels, masses):
#         # Load IL agent
#         cropped_obs_shape = (3 * args.frame_stack, 84, 84)
#         #il_agent = make_il_agent(
#         il_agent = make_il_agent_visual(
#             obs_shape=cropped_obs_shape,
#             action_shape=envs[0].action_space.shape,
#             #dynamics_input_shape=mass.shape[0],
#             args=args)
#         load_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
#         il_agent.load(load_dir, checkpoint)
#         il_agents.append(il_agent)
#
#     return envs, masses, il_agents


def verify_weights(args):
    """Verify if agents indeed share weights"""

    envs, masses, il_agents = setup(args, [0.4, 0.2], ["_0_4", "_0_2"])

    for agt in il_agents:
        #print(agt.actor.encoder.fc.state_dict())
        print(agt.domain_spe.encoder.fc.state_dict())

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

    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

    # 3. Non adapting agent
    reward, _, _ = evaluate_agent(il_agent, env, args, buffer=traj_buffer)
    print('non adapting reward:', int(reward.mean()), ' +/- ', int(reward.std()), ' for label ', label)

    # 4 . Adapting agent
    env = init_env(args, domain[0])
    print(f'Policy Adaptation during Deployment for IL agent of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
    reward, _, _ = evaluate_agent(il_agent, env, args, buffer=traj_buffer, adapt=True)
    print('pad reward:', int(reward.mean()), ' +/- ', int(reward.std()), ' for label ', label)


def test_agents(args):
    il_agents_train, experts, envs, dynamics, buffers, trajs_buffers, stats_expert = setup(args)

    for agent, env, traj, label in zip(il_agents_train, envs, trajs_buffers, ["_0_4", "_0_2", "_0_25", "_0_3"]):

        rewards, _, _ = evaluate_agent(agent, env, args, buffer=traj)
        print(f'For {label} agent : {rewards.mean()} +/- {rewards.std()}')




if __name__ == "__main__":
    args = parse_args()
    test_agents(args)
    