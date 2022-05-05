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
from utils import get_curl_pos_neg, AdaptRecorder
from eval import evaluate, init_env

def compare_agents(args, agent, envs, recorder, video, exp_type, eval_fct = evaluate, reload=True) :

    # Evaluate agent without PAD
    print(f'Evaluating {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
    eval_reward, std = eval_fct(envs, agent, args, video, recorder, reload=reload, exp_type=exp_type)
    print('eval reward:', int(eval_reward), ' +/- ', int(std))

    # Evaluate agent with PAD (if applicable)
    pad_reward = None
    if args.use_inv or args.use_curl or args.use_rot:
        print(f'Policy Adaptation during Deployment of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
        pad_reward, std = eval_fct(envs, agent, args, video, recorder, adapt=True, reload=reload, exp_type=exp_type)
        print('pad reward:', int(pad_reward), ' +/- ', int(std))

def evaluate_seq(envs, agent, args, video, recorder, exp_type, adapt=False, reload = True) :

    print(f'----Evaluating a sequence of environments : {[env._mode for env in envs]}----')
    if adapt :
        print(f'Policy Adaptation during Deployment of {args.work_dir} for {args.pad_num_episodes} episodes)')
    else :
        print(f'Non-adapting agent deployment of {args.work_dir} for {args.pad_num_episodes} episodes)')
    print(f'With pre-trained reload' if reload else f'Without pre-trained reload')
    print("-"*60)

    episode_rewards = []

    def run_episode(env):
        if reload :
            ep_agent = deepcopy(episode_agent)
        else :
            ep_agent = episode_agent
        ep_rew = 0
        obs = env.reset()
        done = False
        losses = []
        step = 0
        rewards = []
        ep_agent.train()

        while not done:
            # Take step
            with utils.eval_mode(ep_agent):
                action = ep_agent.select_action(obs)
            next_obs, reward, done, info, change = env.step(action, rewards)
            ep_rew += reward
            recorder.update(change, reward)

            # Prepare batch of observations
            batch_obs = utils.batch_from_obs(torch.Tensor(obs).cuda(), batch_size=args.pad_batch_size)
            batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
            batch_action = torch.Tensor(action).cuda().unsqueeze(0).repeat(args.pad_batch_size, 1)

            # Adapt using inverse dynamics prediction
            losses.append(ep_agent.update_inv(utils.random_crop(batch_obs), utils.random_crop(batch_next_obs),
                                              batch_action))

            video.record(env, losses)
            obs = next_obs
            step += 1

        return ep_rew


    for i in tqdm(range(args.pad_num_episodes)):

        episode_agent = deepcopy(agent)
        video.init(enabled=True)
        episode_reward = 0

        for env in envs :
            episode_reward += run_episode(env)

        episode_rewards.append(episode_reward)

        recorder.end_episode()
        video.save(f'{args.mode}_pad_{i}.mp4' if adapt else f'{args.mode}_eval_{i}.mp4')

    recorder.save("performance_"+ exp_type, adapt)
    mean, std = np.mean(episode_rewards), np.std(episode_rewards)
    print('pad reward:', int(mean), ' +/- ', int(std))


def main(args):

    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

    # Prepare agent
    # Initialize environment
    args.mode = 'train'
    env = init_env(args)
    assert torch.cuda.is_available(), 'must have cuda enabled'
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=args
    )
    agent.load(model_dir, args.pad_checkpoint)

    # Recorder
    recorder = AdaptRecorder(args.work_dir, args.mode)

    # How does agent behave in test mode in the training environment
    #compare_agents(args, agent, env, recorder, video, exp_type="train", eval_fct=evaluate)

    # How does agent behave when deployed successively in different environment and then back in the training environment
    args.mode = 'color_easy'
    env_color_easy = init_env(args)
    envs = [env_color_easy, env]

    evaluate_seq(envs, agent, args, video, recorder, adapt=True, reload=True, exp_type="reloaded")
    evaluate_seq(envs, agent, args, video, recorder, adapt=True, reload=False, exp_type="not_reloaded")


if __name__ == '__main__':
    args = parse_args()
    main(args)
