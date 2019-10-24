import gym, os
from actor_critic_keras import Agent
from utils import plotLearning
from gym import wrappers
from collections import deque
import numpy as np
import cv2


env_name = 'SpaceInvaders-v0'


if __name__ == '__main__':
    env = gym.make(env_name)
    state_dimension = env.observation_space.shape
    #setto come dimensione quella del reshape
    n_actions = env.action_space.n

    agent = Agent(n_actions=n_actions, input_dims = state_dimension, alpha=0.00001, beta=0.0005, gamma = 0.9)

    score_history = agent.score_history
    num_episodes = 30

    while len(agent.score_history) < num_episodes:
        done = False
        score = 0
        observation = env.reset()
        agent.stack_frames(observation, True)
        #"banale" loop di interazione con gym
        while not done:
            env.render()
            stacked_observation, agent.stacked_frames = agent.stack_frames(observation)
            action = agent.choose_action(stacked_observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(stacked_observation, action, reward, observation_, done)
            observation = observation_
            score = score + reward
        agent.score_history.append(score)

        #salvataggio dello stato dell'apprendimento ogni 10 episodi
        if ((len(agent.score_history) % 10) == 0):
            agent.save(env_name)
        env.close()

        avg_score = np.mean(score_history[-100:])
        avg_10_score = np.mean(score_history[-10:])
        print('episode: ', len(agent.score_history),'score: %.2f' % score,
              'avg last 100 episode score: %.2f' % avg_score, ' avg last 10 episode score: %.2f ' % avg_10_score)

    plotLearning(score_history, filename=env_name, window=100)

