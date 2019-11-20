import gym, os
from advantage_actor_critic import Agent
from utils import plotLearning, plotLosses
from gym import wrappers
from collections import deque
import numpy as np
import cv2
import atari_wrappers as aw


env_name = 'SpaceInvaders-v4'

if __name__ == '__main__':
    # env = aw.FrameStack(aw.NoopResetEnv(aw.ClipRewardEnv(gym.make(env_name)), 35), stack_size)
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    state_dimension = env.observation_space.shape
    #setto come dimensione quella del reshape
    n_actions = env.action_space.n

    agent = Agent(
        n_actions=env.action_space.n, 
        input_dims = env.observation_space.shape, 
        actor_lr=0.00008, 
        critic_lr=0.00008, 
        discount_factor = 0.99, 
        entropy_coefficient=0.01, 
        state = env.reset()[np.newaxis, :],
    )

    score_history = agent.score_history
    num_episodes = 0

    while len(agent.score_history) < num_episodes:
        done = False
        score = 0
        observation = env.reset()
        observation = observation[np.newaxis, :]
        agent.reset_memory()
        step_number = 0
        while not done:
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
                v = agent.get_value(observation_)[0]
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

