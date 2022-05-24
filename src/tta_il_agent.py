import os
import numpy as np

import utils
from arguments import parse_args
from agent.IL_agent import make_il_agent
from eval import init_env
from imitation_learning import evaluate_agent

def main(args):

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
    il_agent.load(load_dir)

    rewards, _, _, feat_vects = evaluate_agent(il_agent, env, args, feat_analysis=True)
    feat_vects = np.array(feat_vects)
    print("Overall stats : {} +/- {}".format(feat_vects.mean(axis=0), feat_vects.std(0)))

if __name__ == "__main__":
    args = parse_args()
    main(args)