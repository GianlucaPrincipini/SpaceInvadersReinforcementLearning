

import gym, os
from a2cnsteps_model import Agent
from utils import plotLearning
from gym import wrappers
from collections import deque
import numpy as np
import cv2


env_name = 'SpaceInvaders-v0'
n_steps = 10
batch_size = 1

def generate_episode(env, agent, state_0, score):
    '''
    Outputs
    =====================================================
    states: list of arrays of states
    actions: list of actions
    rewards: list of rewards
    dones: list of boolean values indicating if an 
        episode completed or not
    next_states: list of arrays of states
    '''
    
    states, actions, rewards, dones, next_states = [], [], [], [], []
    counter = 0
    total_count = batch_size * n_steps
    
    while counter < total_count:
        done = False
        while done == False:
            env.render()
            action = agent.choose_action(state_0)
            state_1, r, done, _ = env.step(action)
            score = score + r
            states.append(agent.preprocess(state_0))
            next_states.append(agent.preprocess(state_1))
            actions.append(action)
            rewards.append(r)
            dones.append(done)
            state_0 = state_1
            
            if done:
                agent.score_history.append(score)
                avg_score = np.mean(agent.score_history[-100:])
                avg_10_score = np.mean(agent.score_history[-10:])
                print('episode: ', len(agent.score_history),'score: %.2f' % score,
                    'avg last 100 episode score: %.2f' % avg_score, ' avg last 10 episode score: %.2f ' % avg_10_score)
                state_0 = env.reset()
                score = 0
                if ((len(agent.score_history) > 0) and ((len(agent.score_history) % 10) == 0)):
                    print("Model saved")
                    agent.save(env_name)
            
            counter += 1
            if counter >= total_count:
                break
    
    return states, actions, rewards, dones, next_states, score


if __name__ == '__main__':
    env = gym.make(env_name)
    state_dimension = env.observation_space.shape
    #setto come dimensione quella del reshape
    n_actions = env.action_space.n
    score = 0
    agent = Agent(n_actions=n_actions, input_dims = state_dimension, alpha=0.00001, beta=0.0005, gamma = 0.9)

    score_history = agent.score_history
    num_episodes = 5000

    state_0 = env.reset()
    while len(agent.score_history) < num_episodes:
        #"banale" loop di interazione con gym
        batch = generate_episode(env, agent, state_0, score)
        score = batch[5]
        agent.learn(batch, env_name)

        #salvataggio dello stato dell'apprendimento ogni 10 episodi
        
    env.close()
    plotLearning(score_history, filename=env_name, window=100)

