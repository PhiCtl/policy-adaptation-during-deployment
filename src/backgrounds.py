import numpy as np
import torch
import os
import itertools
from agent.agent import make_agent
from utils import make_dir, EnvtRecorder
from eval import evaluate, init_env
from video import VideoRecorder
from arguments import parse_args


def main(args):

    # Generate lists of modifs
    hue = [0.1, 0.2, 0.3, 0.4, 0.5]
    contrast = [0.5, 1.5]
    brightness = [0.5, 1.5]
    combinations = list(itertools.product(hue, contrast, brightness))

    # create env
    env = init_env(args)

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)
    model_dir = make_dir(os.path.join(args.work_dir, 'model'))
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=args
    )
    agent.load(model_dir, args.pad_checkpoint)

    # Create recorder
    video = VideoRecorder(None, height=448, width=448)
    recorder = EnvtRecorder(args.work_dir, args.mode)

    # Apply to background and evaluate on non PAD
    for i in range(4) :
        bg = "video" + str(i) + "_frame.jpeg"
        env.load_background(bg)
        recorder.load_background(bg)
        print(f'Evaluating {bg} for {args.pad_num_episodes} episodes (mode: {args.mode})')

        for h, c, b in combinations:

            env.change_background({"b" : b, "h": h, "c" : c})
            recorder.load_change({"b" : b, "h": h, "c" : c})
            eval_reward, std = evaluate(env, agent, args, video, recorder)
            print("Params h {} b {} c {} mean {} std {}".format(h,b,c,eval_reward, std))

    # Save on recorder
    recorder.close()

if __name__ == '__main__':
	args = parse_args()
	main(args)
