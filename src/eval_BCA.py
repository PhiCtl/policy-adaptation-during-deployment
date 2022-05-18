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
from utils import AdaptRecorder

def init_env(args):
    utils.set_seed_everywhere(args.seed)
    return make_pad_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        mode=args.mode,
        dependent=args.dependent,
        threshold=args.threshold,
        window=args.window
    )

def prepare_BCA(env, agent, buffer, num_episodes) :
    """Evaluate an agent without adaptation in training environment"""
    print("Fill in source buffer")
    print("-"*60)

    for i in tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        step = 0
        rewards = []

        while not done:

            with utils.eval_mode(agent):
                action = agent.select_action(obs)

            next_obs, reward, done, _, _,_ = env.step(action, rewards)
            done_bool = 0 if step + 1 == env._max_episode_steps else float(done)
            buffer.add(obs, action, reward, next_obs, done_bool)
            step += 1

    print("Source buffer filled in !")
    print("-" * 60)

def evaluate(env, agent, clone, buffer, args, video, recorder, exp_type="", adapt=False, bca=False, reload=False):
    episode_rewards = []

    for i in tqdm(range(args.pad_num_episodes)):
        ep_agent = deepcopy(agent)  # make a new copy

        video.init(enabled=True)
        obs = env.reset()
        done = False
        episode_reward = 0
        losses = []
        step = 0
        rewards = []
        ep_agent.train()

        while not done:

            # Take step
            with utils.eval_mode(ep_agent):
                action = ep_agent.select_action(obs)
            next_obs, reward, done, info, change, has_changed = env.step(action, rewards)
            episode_reward += reward
            recorder.update(change, reward)

            # Make self-supervised update if flag is true
            if adapt:

                if args.use_inv:  # inverse dynamics model

                    # Prepare batch of observations
                    batch_obs = utils.batch_from_obs(torch.Tensor(obs).cuda(), batch_size=args.pad_batch_size)
                    batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
                    batch_action = torch.Tensor(action).cuda().unsqueeze(0).repeat(args.pad_batch_size, 1)

                    # Adapt using inverse dynamics prediction
                    losses.append(ep_agent.update_inv(utils.random_crop(batch_obs), utils.random_crop(batch_next_obs),
                                                      batch_action))

                if bca:
                    # Adapt using KL divergence loss and train Actor-Critic network
                    agent.update_actor_and_alpha(obs, bca_loss=bca, buffer=buffer, clone=clone, update_alpha=False)

            # TODO remove losses as they're not updated with actor loss and not printed either
            video.record(env, losses)
            obs = next_obs
            step += 1

            if has_changed and reload :
                ep_agent = deepcopy(agent)

        video.save(f'{args.mode}_pad_{i}.mp4' if adapt else f'{args.mode}_eval_{i}.mp4')
        episode_rewards.append(episode_reward)
        recorder.end_episode()

    recorder.save("performance_"+exp_type, adapt)
    return np.mean(episode_rewards), np.std(episode_rewards)

def main(args):

    # Initialize environments : source and target
    env = init_env(args) # target
    training_env = make_pad_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        mode='train',
        dependent=False) # source

    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

    # Prepare agents
    assert torch.cuda.is_available(), 'must have cuda enabled'
    cropped_obs_shape = (3 * args.frame_stack, 84, 84)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=args
    )
    agent.load(model_dir, args.pad_checkpoint)

    # Teacher agent
    expert = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=args
    )
    expert.load(model_dir, args.pad_checkpoint)

    # Collect observations from source environment
    replay_buffer = utils.ReplayBuffer(
        obs_shape=training_env.observation_space.shape,
        action_shape=training_env.action_space.shape,
        capacity=args.train_steps,
        batch_size=args.pad_batch_size
    )
    prepare_BCA(training_env, expert, replay_buffer, args.pad_num_episodes)

    # Recorder
    recorder = AdaptRecorder(args.work_dir, args.mode)

    # Evaluate agent without PAD
    print(f'Evaluating {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode})')
    eval_reward, std = evaluate(env, agent, None, None, args, video, recorder, adapt=False, bca=False, exp_type="eval")
    print('eval reward:', int(eval_reward), ' +/- ', int(std))

    # Evaluate agent with PAD (if applicable)
    if args.use_inv or args.use_curl or args.use_rot:
        env = init_env(args)
        print( f'Policy Adaptation during Deployment of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode}) with BCA')
        pad_reward, std = evaluate(env, agent, expert, replay_buffer, args, video, recorder, adapt=True, bca=True, exp_type="bca")
        print('pad reward:', int(pad_reward), ' +/- ', int(std))

        # env = init_env(args)
        # print( f'Policy Adaptation during Deployment of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode}) without BCA')
        # pad_reward, std = evaluate(env, agent, clone_agent, replay_buffer, args, video, recorder, adapt=True, bca=False, exp_type="normal")
        # print('pad reward:', int(pad_reward), ' +/- ', int(std))

        env = init_env(args)
        print(f'Policy Adaptation during Deployment of {args.work_dir} for {args.pad_num_episodes} episodes (mode: {args.mode}) with reload')
        pad_reward, std = evaluate(env, agent, None, None, args, video, recorder, adapt=True, bca=False, exp_type="reloaded", reload=True)
        print('pad reward:', int(pad_reward), ' +/- ', int(std))



    print(f' Threshold {args.threshold} Window size {args.window}')

if __name__ == "__main__" :
    args = parse_args()
    main(args)

