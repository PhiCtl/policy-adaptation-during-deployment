import numpy as np
import torch
import os
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import utils
from video import VideoRecorder

from arguments import parse_args
from env.wrappers import make_pad_env
from agent.agent import make_agent
from utils import get_curl_pos_neg
from recorder import AdaptRecorder


def evaluate(env, agent, args, buffer=None, video=None, recorder=None, adapt=False, reload=False, exp_type=""):
    """Evaluate an agent, optionally adapt using PAD"""
    episode_rewards = []

    for i in tqdm(range(args.pad_num_episodes)):
        ep_agent = deepcopy(agent)  # make a new copy

        if args.use_curl:  # initialize replay buffer for CURL
            replay_buffer = utils.ReplayBuffer(
                obs_shape=env.observation_space.shape,
                action_shape=env.action_space.shape,
                capacity=args.train_steps,
                batch_size=args.pad_batch_size
            )
            
        if video: video.init(enabled=True)
        obs = env.reset()
        done = False
        episode_reward = 0
        losses = []
        step = 0
        rewards = []
        ep_agent.train()

        while not done:
            # Take step
            traj = None if buffer is None else buffer.sample_traj()
            with utils.eval_mode(ep_agent):
                action = ep_agent.select_action(obs, traj)
            next_obs, reward, done, info, change, has_changed = env.step(action, rewards)
            episode_reward += reward
            if recorder : recorder.update(change, reward)

            # Make self-supervised update if flag is true
            if adapt:
                if args.use_rot:  # rotation prediction

                    # Prepare batch of cropped observations
                    batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
                    batch_next_obs = utils.random_crop(batch_next_obs)

                    # Adapt using rotation prediction
                    losses.append(ep_agent.update_rot(batch_next_obs))

                if args.use_inv:  # inverse dynamics model

                    # Prepare batch of observations
                    batch_obs = utils.batch_from_obs(torch.Tensor(obs).cuda(), batch_size=args.pad_batch_size)
                    batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
                    batch_action = torch.Tensor(action).cuda().unsqueeze(0).repeat(args.pad_batch_size, 1)

                    # Adapt using inverse dynamics prediction
                    if buffer:
                        b_o1 = utils.batch_from_obs(traj[0], batch_size=args.pad_batch_size)
                        b_o2 = utils.batch_from_obs(traj[2], batch_size=args.pad_batch_size)
                        b_o3 = utils.batch_from_obs(traj[4], batch_size=args.pad_batch_size)
                        b_a1 = traj[1].repeat(args.pad_batch_size, 1)
                        b_a2 = traj[3].repeat(args.pad_batch_size, 1)
                        losses.append(ep_agent.update_inv(utils.random_crop(batch_obs), utils.random_crop(batch_next_obs),
                                                      batch_action, [b_o1, b_a1, b_o2, b_a2, b_o3]))
                    else :
                        losses.append(ep_agent.update_inv(utils.random_crop(batch_obs), utils.random_crop(batch_next_obs),
                                                      batch_action))

                if args.use_curl:  # CURL

                    # Add observation to replay buffer for use as negative samples
                    # (only first argument obs is used, but we store all for convenience)
                    replay_buffer.add(obs, action, reward, next_obs, True)

                    # Prepare positive and negative samples
                    obs_anchor, obs_pos = get_curl_pos_neg(next_obs, replay_buffer)

                    # Adapt using CURL
                    losses.append(ep_agent.update_curl(obs_anchor, obs_pos, ema=True))

            if video: video.record(env, losses)
            obs = next_obs
            step += 1

            if has_changed and reload:
                ep_agent = deepcopy(agent)

        if video: video.save(f'{args.mode}_pad_{i}.mp4' if adapt else f'{args.mode}_eval_{i}.mp4')
        episode_rewards.append(episode_reward)
        if recorder: recorder.end_episode()

    if recorder: recorder.save("performance_"+ exp_type, adapt)
    return np.mean(episode_rewards), np.std(episode_rewards)


def init_env(args, mass=None, force=None, seed=None):
    utils.set_seed_everywhere(args.seed if seed is None else seed)
    return make_pad_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        mode=args.mode,
        dependent=args.dependent,
        window=args.window,
        mass= mass if mass else args.cart_mass,
        force=force if force else args.force_walker
    )


def main(args):
    # Initialize environment
    env = init_env(args)

    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
    recorder = AdaptRecorder(args.work_dir, args.mode)

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=args
    )
    agent.load(model_dir, args.pad_checkpoint)


    # Evaluate agent without PAD
    print(f'Evaluating {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
    eval_reward, std = evaluate(env, agent, args, video = video, recorder = recorder)
    print('eval reward:', int(eval_reward), ' +/- ', int(std))

    # Evaluate agent with PAD (if applicable)
    pad_reward = None
    if args.use_inv or args.use_curl or args.use_rot:
        env = init_env(args)
        print( f'Policy Adaptation during Deployment of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
        pad_reward, std = evaluate(env, agent, args, video = video, recorder = recorder, adapt=True, exp_type="pad")
        print('pad reward:', int(pad_reward), ' +/- ', int(std))

        # env = init_env(args)
        # print(
        #     f'Policy Adaptation during Deployment of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
        # pad_reward, std = evaluate(env, agent, args, video, recorder, adapt=True, reload=True, exp_type="reloaded_pad")
        # print('pad reward:', int(pad_reward), ' +/- ', int(std))

    # Save results
    results_fp = os.path.join(args.work_dir, f'pad_{args.mode}.pt')
    torch.save({
        'args': args  #,
        #'eval_reward': eval_reward,
        #'pad_reward': pad_reward
    }, results_fp)
    print('Saved results to',results_fp)


if __name__ == '__main__':
    args = parse_args()
    main(args)
