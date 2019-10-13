import gym, os
from actor_critic_keras import Agent
from utils import plotLearning
from gym import wrappers
import numpy as np

env_name = 'SpaceInvaders-v0'

if __name__ == '__main__':
    env = gym.make(env_name)
    state_dimension = env.observation_space.shape
    n_actions = env.action_space.n

    agent = Agent(n_actions=n_actions, input_dims = state_dimension, alpha=0.00001, beta=0.00005, gamma = 0.99, is_ram=False)

    score_history = []
    num_episodes = 100

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, observation_, done)
            observation = observation_
            score += reward
        env.close()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('episode: ', i,'score: %.2f' % score,
              'avg score %.2f' % avg_score)

    plotLearning(score_history, filename=env_name, window=100)