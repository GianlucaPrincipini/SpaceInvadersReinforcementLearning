import gym, os
from convolutional_advantage_actor_critic import Agent
from utils import plotLearning, plotLosses
from gym import wrappers
from collections import deque
import numpy as np
import cv2
import atari_wrappers as aw
from time import sleep


env_name = 'SpaceInvaders-v0'
n_env = 4
stack_size = 4

if __name__ == '__main__':
    env = aw.FrameStack(aw.NoopResetEnv(aw.ClipRewardEnv(gym.make(env_name)), 35), stack_size)
    state_dimension = env.observation_space.shape
    #setto come dimensione quella del reshape
    n_actions = env.action_space.n

    agent = Agent(n_actions=n_actions, 
        input_dims = state_dimension, 
        stack_size = stack_size, 
        actor_lr=0.00003, 
        critic_lr=0.00003, 
        discount_factor = 0.9, 
        entropy_coefficient=0.02, 
        state = env.reset()[np.newaxis, :],
        n_env=n_env,
        # env_name = env_name
    )

    score_history = agent.score_history
    num_episodes = 12000

    
    while len(agent.score_history) < num_episodes:
        #"banale" loop di interazione con gym
        i = 0
        last_value = []
        while i < n_env:
            step_number = 0
            score = 0
            done = False
            agent.reset_memory()
            observation = env.reset()
            observation = observation[np.newaxis, :]
            while not done:
                env.render()
                action = agent.choose_action(observation)
                # sleep(1)
                observation_, reward, done, info = env.step(action)
                observation_ = observation_[np.newaxis, :]
                step = [step_number, observation, observation_, reward, done]
                agent.remember(i, step)
                step_number = step_number + 1
                score = score + reward
                observation = observation_
                if done:
                    i = i + 1
                    last_value.append(agent.get_value(observation_)[0])
                    agent.score_history.append(score)
                    print('episode: ', len(agent.score_history),'score: %.2f' % score)

        if i == n_env:
            # L'addestramento avviene alla fine di ogni episodio
            agent.train_by_episode(last_value=last_value)

        avg_score = np.mean(score_history[-100:])
        avg_10_score = np.mean(score_history[-10:])
        print('avg last 100 episode score: %.2f' % avg_score, ' avg last 10 episode score: %.2f ' % avg_10_score)

        #salvataggio dello stato dell'apprendimento ogni 10 episodi
        if ((len(agent.score_history) % 10) == 0):
            agent.save(env_name)


    plotLearning(score_history, filename=env_name, window=100)
    plotLosses(agent.actor_losses, filename=env_name + '_actor_losses', window=100)
    plotLosses(agent.critic_losses, filename=env_name + '_critic_losses', window=100)

