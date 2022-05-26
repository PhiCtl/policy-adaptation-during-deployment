import os
import numpy as np

import utils
from video import VideoRecorder
from arguments import parse_args
from agent.IL_agent import make_il_agent
from eval import init_env, evaluate
from imitation_learning import evaluate_agent

def verify_weights(args):
    # Load env for a give mass
    envs = []
    masses = []
    for label in [0.3, 0.2]:
        env = init_env(args, label)
        masses.append(env.get_masses())
        envs.append(env)

    il_agents = []
    for label, mass in zip(["_0_3", "_0_2"], masses):
        # Load IL agent
        cropped_obs_shape = (3 * args.frame_stack, 84, 84)
        il_agent = make_il_agent(
            obs_shape=cropped_obs_shape,
            action_shape=envs[0].action_space.shape,
            dynamics_input_shape=mass.shape[0],
            args=args)
        load_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model'))
        il_agent.load(load_dir, "0")
        il_agents.append(il_agent)

    for agt in il_agents:
        print(agt.actor.encoder.fc.state_dict())
        print(agt.ss_encoder.fc.state_dict())


def main(args):
    domain = args.domain_test
    label = args.label
    rd = args.rd

    """Performs IL agent test time adaptation"""

    # 1. Load agent

    # Load environment
    env = init_env(args, domain) # domain is target cart mass value
    mass = env.get_masses()
    # Load IL agent
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)
    il_agent = make_il_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        dynamics_input_shape=mass.shape[0],
        args=args)
    load_dir = utils.make_dir(os.path.join(args.save_dir, label, 'model')) # Il agent trained on specific domain
    il_agent.load(load_dir, "final") # model checkpoint (5 / final = 10 dagger iterations)

    # Initialize feature vector either at random either with domain_specific feature vector
    if rd :
        init = np.random.rand(args.dynamics_output_shape)
    else :
        init = il_agent.extract_feat_vect(mass)
    il_agent.init_feat_vect(init)

    # 2. Prepare test time evaluation
    recorder = utils.AdaptRecorder(args.work_dir, args.mode)

    # 3. Non adapting agent
    reward, std = evaluate(env, il_agent, args, video=None, recorder=recorder, adapt=False, exp_type="il")
    print('non adapting reward:', int(reward), ' +/- ', int(std))

    # 4 . Adapting agent
    env = init_env(args, domain)
    print(f'Policy Adaptation during Deployment for IL agent of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
    pad_reward, std = evaluate(env, il_agent, args, video=None, recorder=recorder, adapt=True, exp_type="il_adapt")
    print('pad reward:', int(pad_reward), ' +/- ', int(std))


if __name__ == "__main__":
    args = parse_args()
    main(args)
    