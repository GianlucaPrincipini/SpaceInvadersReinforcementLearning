from keras import backend as K
from keras.layers import Dense, Activation, Input, Conv2D, Flatten, LSTM, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import TimeDistributed
import tensorflow as tf
import pickle
import numpy as np
from math import log
import cv2
from collections import deque
import matplotlib.pyplot as plt

stack_size = 4

class Agent(object):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=4,
                 layer1_size=256, layer2_size=256, input_dims=8, n_steps = 10, env_name = ''):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.n_steps = n_steps
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.score_history = []
        self.stacked_frames = deque([np.zeros((76, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        self.actor, self.critic, self.policy = self.build_actor_critic_network(env_name)
        self.action_space = [i for i in range(n_actions)]

    
    def calc_rewards(self, batch):
        '''
        Inputs
        =====================================================
        batch: tuple of state, action, reward, done, and
            next_state values from generate_episode function
        
        Outputs
        =====================================================
        R: np.array of discounted rewards
        G: np.array of TD-error
        '''
        
        states, actions, rewards, dones, next_states = batch
        # Convert values to np.arrays
        rewards = np.array(rewards)
        print(states[0].shape)
        states = np.vstack(states)[np.newaxis, :]
        next_states = np.vstack(next_states)[np.newaxis, :]
        actions = np.array(actions)
        dones = np.array(dones)
        
        total_steps = len(rewards)
        
        print(states.shape)
        state_values = self.critic.predict(states)
        next_state_values = self.critic.predict(next_states)
        next_state_values[dones] = 0
        
        R = np.zeros_like(rewards, dtype=np.float32)
        G = np.zeros_like(rewards, dtype=np.float32)
        
        for t in range(total_steps):
            last_step = min(self.n_steps, total_steps - t)
            
            # Look for end of episode
            check_episode_completion = dones[t:t + last_step]
            if check_episode_completion.size > 0:
                if True in check_episode_completion:
                    next_ep_completion = np.where(check_episode_completion == True)[0][0]
                    last_step = next_ep_completion
            
            # Sum and discount rewards
            R[t] = sum([rewards[t + n:t + n + 1]* self.gamma **n for
                        n in range(last_step)])
        
        if total_steps > self.n_steps:
            R[:total_steps - self.n_steps] += next_state_values[self.n_steps:]
            
        G = R - state_values
        return R, G


    #funzione di preprocess: rendiamo i frame monocromatici e
    #tagliamo tutte le parti di immagine inutili all'apprendimento e normalizziamo tutti i valori a 1
    def preprocess(self, observation):
        retObs = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        retObs = retObs[9:102,:]
        ret, retObs = cv2.threshold(retObs,1,255,cv2.THRESH_BINARY)
        return np.reshape(retObs / 255,(93,84,1))

    def build_actor_critic_network(self, env_name):
        #creazione della struttura delle nostre due reti
        input = Input(shape=(93, 84, 1))
        head = Conv2D(64, kernel_size=(3, 3), activation='relu')(input)
        conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(head)
        input_tail_network = Flatten()(conv1)
        dense1 = Dense(self.fc1_dims, activation='relu')(input_tail_network)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)


        actor = Model(input=[input], output=[probs])
        critic = Model(input=[input], output=[values])
        critic.summary()
        policy = Model(input=[input], output=[probs])


        if (env_name != ''):
            actor.load_weights(env_name + '_actor.h5')        
            critic.load_weights(env_name + '_critic.h5')
            with open (env_name + '_scores.dat', 'rb') as fp:
                self.score_history = pickle.load(fp)
            
        self.entropy = 0 

        def custom_entropy_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-5, 1-1e-5)
            log_lik = K.log(out)*y_true + 0.01 * self.entropy
            loss = K.sum(-log_lik)
            return loss

        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_entropy_loss)
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        return actor, critic, policy

    def choose_action(self, observation):
        state = self.preprocess(observation)[np.newaxis, :]
        probabilities = self.policy.predict(state)[0] 
        action = np.random.choice(self.action_space, p=probabilities)

        # print(probabilities)
        #Il clipping delle probabilità evita il logaritmo -inf
        self.entropy = - (tf.math.reduce_sum(probabilities * tf.math.log(tf.clip_by_value(probabilities,1e-5,1.0 - 1e-5))))
        return action

    def learn(self, batch):
        # state = current_stacked_state[np.newaxis,:]
        # target = np.zeros((1, 1))
        # advantages = np.zeros((1, self.n_actions))
        # s_, dump = self.stack_frames(state_)
        # state_ = s_[np.newaxis,:]

        # value = self.critic.predict(state)[0]
        # next_value = self.critic.predict(state_)[0]

        # if done:
        #     advantages[0][action] = reward - value
        #     target[0][0] = reward
        # else:
        #     advantages[0][action] = reward + self.gamma * next_value - value
        #     target[0][0] = reward + self.gamma * next_value

        #advantages = np.reshape(advantages, (1, advantages.shape[0], advantages.shape[1]))
        #print(np.reshape(advantages, (advantages.shape[1])))
        #target = np.reshape(target, (1, target.shape[1]))
        R, G = self.calc_rewards(batch)
        states = np.vstack(batch[0])[np.newaxis, :]
        self.actor.fit(states, G, epochs=1, verbose=0)
        self.critic.fit(states, R, epochs=1, verbose=0)

        # state_ = s_[np.newaxis,:]
        # critic_value_ = self.critic.predict(state_)
        # critic_value = self.critic.predict(state)


        # target = reward + self.gamma*critic_value_*(1-int(done))
        # delta =  target - critic_value

        # actions = np.zeros([1, self.n_actions])
        # actions[np.arange(1), action] = 1

        # # print(actions)
        # self.actor.fit([state, delta], actions, verbose=0)

        # self.critic.fit(state, target, verbose=0)

    def save(self, envName):
        self.actor.save_weights(envName + '_actor.h5')        
        self.critic.save_weights(envName + '_critic.h5')
        with open(envName + '_scores.dat', 'wb') as fp:
            pickle.dump(self.score_history, fp)

        