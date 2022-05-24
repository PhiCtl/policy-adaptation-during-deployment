import os
import numpy as np

import utils
from video import VideoRecorder
from arguments import parse_args
from agent.IL_agent import make_il_agent
from eval import init_env, evaluate
from imitation_learning import evaluate_agent

def stays_constant(args):

    # 1. test whether feature vector is constant for inference

    # Load env for a give mass
    env = init_env(args, 0.3)
    mass = env.get_masses()

    # Load IL agent
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)
    il_agent = make_il_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        dynamics_input_shape=mass.shape[0],
        args=args)
    load_dir = utils.make_dir(os.path.join(args.save_dir, "_0_3", 'model'))
    il_agent.load(load_dir, "12")

    rewards, _, _, feat_vects = evaluate_agent(il_agent, env, args, feat_analysis=True)
    feat_vects = np.array(feat_vects)
    print("Overall stats : {} +/- {}".format(feat_vects.mean(axis=0), feat_vects.std(0)))

def main(args):

    # 1. Load agent

    # Load environment
    env = init_env(args, 0.35) # tried with out of range value
    mass = env.get_masses()

    # Load IL agent
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)
    il_agent = make_il_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        dynamics_input_shape=mass.shape[0],
        args=args)
    load_dir = utils.make_dir(os.path.join(args.save_dir, "_0_3", 'model'))
    il_agent.load(load_dir, "12")

    # Initialize feature vector
    il_agent.init_feat_vect(il_agent.extract_feat_vect(0.35))

    # 2. Prepare test time evaluation
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
    recorder = utils.AdaptRecorder(args.work_dir, args.mode)

    # 3. Non adapting agent
    reward, std = evaluate(env, il_agent, args, video, recorder, adapt=False, exp_type="il")
    print('non adapting reward:', int(reward), ' +/- ', int(std))

    # 4 . Adapting agent
    env = init_env(args, 0.35)
    print(f'Policy Adaptation during Deployment for IL agent of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
    pad_reward, std = evaluate(env, il_agent, args, video, recorder, adapt=True, exp_type="il_adapt")
    print('pad reward:', int(pad_reward), ' +/- ', int(std))


if __name__ == "__main__":
    args = parse_args()
    main(args)