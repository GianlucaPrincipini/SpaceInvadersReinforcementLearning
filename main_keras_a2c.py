import gym, os
from advantage_actor_critic import Agent
from utils import plotLearning
from gym import wrappers
from collections import deque
import numpy as np
import cv2
import atari_wrappers as aw


env_name = 'SpaceInvaders-v4'

stack_size = 4

if __name__ == '__main__':
    #env = aw.FrameStack(aw.NoopResetEnv(aw.ClipRewardEnv(gym.make(env_name)), 35), stack_size)
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    state_dimension = env.observation_space.shape
    #setto come dimensione quella del reshape
    n_actions = env.action_space.n

    agent = Agent(n_actions=n_actions, 
        input_dims = state_dimension, 
        stack_size = stack_size, 
        actor_lr=0.00001, 
        critic_lr=0.00001, 
        discount_factor = 1, 
        entropy_coefficient=0, 
        state = env.reset()[np.newaxis, :],
        # env_name = env_name
    )

    score_history = agent.score_history
    num_episodes = 3200

    while len(agent.score_history) < num_episodes:
        done = False
        score = 0
        step_number = 0
        observation = env.reset()
        observation = observation[np.newaxis, :]
        agent.reset_memory()
        #"banale" loop di interazione con gym
        while not done:
            agent.choose_action(observation)
            env.render()
            action = agent.choose_action(observation)
            # print(action[0])
            observation_, reward, done, info = env.step(action[0])
            observation_ = observation_[np.newaxis, :]

            step = [step_number, observation, observation_, reward, done]
            agent.remember(step)

            score = score + reward
            # agent.learn(observation, action, reward, observation_, done)
            observation = observation_

            if done:
                # for REINFORCE, REINFORCE with baseline, and A2C
                # we wait for the completion of the episode before 
                # training the network(s)
                # last value as used by A2C
                v = 0 
                agent.train_by_episode(last_value=v)
        agent.score_history.append(score)

        #salvataggio dello stato dell'apprendimento ogni 10 episodi
        if ((len(agent.score_history) % 10) == 0):
            agent.save(env_name)

        avg_score = np.mean(score_history[-100:])
        avg_10_score = np.mean(score_history[-10:])
        print('episode: ', len(agent.score_history),'score: %.2f' % score,
              'avg last 100 episode score: %.2f' % avg_score, ' avg last 10 episode score: %.2f ' % avg_10_score)

    env.close()
    plotLearning(score_history, filename=env_name, window=100)

