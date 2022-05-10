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
from eval_BCA import prepare_BCA

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

def evaluate_seq(envs, agent, args, video, recorder, exp_type, clone=None, buffer=None, adapt=False, reload = True, bca=False) :

    print(f'----Evaluating a sequence of environments : {[env._mode for env in envs]}----')
    if adapt :
        print(f'Policy Adaptation during Deployment of {args.work_dir} for {args.pad_num_episodes} episodes)')
    else :
        print(f'Non-adapting agent deployment of {args.work_dir} for {args.pad_num_episodes} episodes)')
    print(f'With pre-trained reload' if reload else f'Without pre-trained reload')
    print(f'With bca' if bca else f'Without bca')
    print("-"*60)

    assert(not (bca and reload)), "Either reload or bca mode is allowed"
    episode_rewards = []

    def run_episode(env):
        if reload:
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
            next_obs, reward, done, info, change, _ = env.step(action, rewards)
            ep_rew += reward
            recorder.update(change, reward)

            # Prepare batch of observations
            batch_obs = utils.batch_from_obs(torch.Tensor(obs).cuda(), batch_size=args.pad_batch_size)
            batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
            batch_action = torch.Tensor(action).cuda().unsqueeze(0).repeat(args.pad_batch_size, 1)

            # Adapt using inverse dynamics prediction
            losses.append(ep_agent.update_inv(utils.random_crop(batch_obs), utils.random_crop(batch_next_obs),
                                              batch_action))

            if bca:
                # Adapt using KL divergence loss and train Actor-Critic network
                agent.update_actor_and_alpha(obs, bca_loss=bca, buffer=buffer, clone=clone)

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

    # Initialize environments
    training_env = make_pad_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        mode='train',
        dependent=False)

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=training_env.action_space.shape,
        args=args
    )
    agent.load(model_dir, args.pad_checkpoint)
    clone = deepcopy(agent)

    # Collect observations from source environment
    replay_buffer = utils.ReplayBuffer(
        obs_shape=training_env.observation_space.shape,
        action_shape=training_env.action_space.shape,
        capacity=args.train_steps,
        batch_size=args.pad_batch_size
    )
    prepare_BCA(training_env, clone, replay_buffer, args.pad_num_episodes)

    # Recorder
    recorder = AdaptRecorder(args.work_dir, args.mode)

    # How does agent behave in test mode in the training environment
    #compare_agents(args, agent, env, recorder, video, exp_type="train", eval_fct=evaluate)

    # How does agent behave when deployed successively in different environment and then back in the training environment
    args.mode = 'color_easy'
    env_color_easy = init_env(args)
    envs = [env_color_easy, training_env]
    evaluate_seq(envs, agent, args, video, recorder, adapt=True, reload=True, exp_type="reloaded")


    env_color_easy = init_env(args)
    envs = [env_color_easy, training_env]
    evaluate_seq(envs, agent, args, video, recorder, buffer=replay_buffer, clone=clone, adapt=True, reload=False, exp_type="bca", bca=True)

    env_color_easy = init_env(args)
    envs = [env_color_easy, training_env]
    evaluate_seq(envs, agent, args, video, recorder, buffer=replay_buffer, clone=clone, adapt=True, reload=False, exp_type="normal", bca=False)

    # How does it behave if deployed successively in different envt with and without weight reloading ?
    # Evaluate agent with weight reloading
    # print(f'Evaluating {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode}) with reloading')
    # eval_reward, std = evaluate(env, agent, args, video, recorder, adapt = True, reload=True, exp_type="reloaded")
    # print('Reloaded reward:', int(eval_reward), ' +/- ', int(std))
    #
    # # Evaluate agent with PAD (if applicable)
    # print(f'Policy Adaptation during Deployment of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode}) without reloading')
    # pad_reward, std = evaluate(env, agent, args, video, recorder, adapt=True, reload=False, exp_type="not_reloaded")
    # print('pad reward:', int(pad_reward), ' +/- ', int(std))

if __name__ == '__main__':
    args = parse_args()
    main(args)
