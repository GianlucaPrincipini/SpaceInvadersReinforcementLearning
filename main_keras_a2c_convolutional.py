import gym, os
from convolutional_advantage_actor_critic import Agent
from utils import plotLearning, plotLosses
from gym import wrappers
from collections import deque
import numpy as np
import cv2
import atari_wrappers as aw


env_name = 'Breakout-v4'

stack_size = 3

if __name__ == '__main__':
    env = aw.FrameStack(aw.TimeLimit(aw.FireResetEnv(gym.make(env_name)), 2000), stack_size)
    state_dimension = env.observation_space.shape
    #setto come dimensione quella del reshape
    n_actions = env.action_space.n

    agent = Agent(n_actions=n_actions, 
        input_dims = state_dimension, 
        stack_size = stack_size, 
        actor_lr=0.0005, 
        critic_lr=0.005 , 
        discount_factor = 0.99, 
        entropy_coefficient=0.02, 
        state = env.reset()[np.newaxis, :],
        # env_name = env_name
    )

    score_history = agent.score_history
    num_episodes = 5000

    while len(agent.score_history) < num_episodes:
        done = False
        score = 0
        observation = env.reset()
        observation = observation[np.newaxis, :]
        agent.reset_memory()
        #"banale" loop di interazione con gym
        step_number = 0
        while not done:
            agent.choose_action(observation)
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = observation_[np.newaxis, :]

            step = [step_number, observation, observation_, reward, done]
            agent.remember(step)
            step_number = step_number + 1
            score = score + reward
            observation = observation_
            
            if done:
                # L'addestramento avviene alla fine di ogni episodio
                v = 0 if reward > 0 else agent.get_value(observation_)[0]
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
    plotLosses(agent.actor_losses, filename=env_name + '_actor_losses', window=100)
    plotLosses(agent.critic_losses, filename=env_name + '_critic_losses', window=100)

