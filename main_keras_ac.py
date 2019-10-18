import gym, os
from actor_critic_keras import Agent
from utils import plotLearning
from gym import wrappers
from collections import deque
import numpy as np
import cv2


def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))


stack_size = 4

def stak_frames(stacked_frames, state, is_newEpisonde):
    frame = state
    if is_newEpisonde:
        #Clear our stack
        stacked_frames = deque([np.zeros((110, 84), dtype= np.int) for i in range(stack_size)], maxlen=4)

        #since qu're in a new episode copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

env_name = 'SpaceInvaders-v0'

if __name__ == '__main__':
    env = gym.make(env_name)
    state_dimension = env.observation_space.shape
    #setto come dimensione quella del reshape
    #state_dimension = (84,84,1)
    n_actions = env.action_space.n

    agent = Agent(n_actions=n_actions, input_dims = state_dimension, alpha=0.00001, beta=0.00005, gamma = 0.99, is_ram=False)

    score_history = agent.score_history
    num_episodes = 5000

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
            score = score + reward
        agent.score_history.append(score)
        agent.save(env_name)
        env.close()

        avg_score = np.mean(score_history[-100:])
        print('episode: ', len(agent.score_history),'score: %.2f' % score,
              'avg score %.2f' % avg_score)

    plotLearning(score_history, filename=env_name, window=100)

