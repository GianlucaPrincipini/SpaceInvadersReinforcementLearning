from keras import backend as K
from keras.layers import Dense, Activation, Input, Conv2D, Flatten, LSTM, Dropout
from keras.models import Model, load_model
from keras.initializers import Orthogonal
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import TimeDistributed
import tensorflow as tf
import sys
import pickle
import numpy as np
from math import log
import cv2
from collections import deque
import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, actor_lr, critic_lr, stack_size = 4, discount_factor=0.99, n_actions=4,
                 layer1_size=128, layer2_size=128, input_dims=8, entropy_coefficient = 0.01, env_name = ''):
        self.discount_factor = discount_factor
        self.stack_size = stack_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.input_dims = input_dims
        self.entropy_coefficient = entropy_coefficient
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.score_history = []
        self.actor, self.critic, self.policy = self.build_actor_critic_network(env_name)
        self.action_space = [i for i in range(n_actions)]

    def getEntropia():
        return self.entropy

    #funzione di preprocess: rendiamo i frame monocromatici e
    #tagliamo tutte le parti di immagine inutili all'apprendimento e normalizziamo tutti i valori a 1
    def preprocess(self, observation):
        retObs = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        retObs = retObs[9:102,:]
        ret, retObs = cv2.threshold(retObs,1,255,cv2.THRESH_BINARY)
        # plt.imshow(np.squeeze(retObs))
        # plt.show()
        return np.reshape(retObs / 255,(93,84,1))

    def build_actor_critic_network(self, env_name):
        #creazione della struttura delle nostre due reti
        entropia = Input(shape = [1])
        advantage = Input(shape = [1])
        input = Input(shape=(93, 84, self.stack_size))
        head =  Conv2D(32, kernel_size=(8, 8), strides=2, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(input)
        conv1 = Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(head)
        conv2 = Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(conv1)
        input_tail_network = Flatten()(conv2)
        # input_tail_network = Input(shape=self.input_dims)
        # input = input_tail_network
        dense_actor_1 =     Dense(self.fc1_dims, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(input_tail_network)
        dense_actor_2 =     Dense(self.fc1_dims, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(dense_actor_1)
        dense_critic_1 =    Dense(self.fc2_dims, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(input_tail_network)
        dense_critic_2 =    Dense(self.fc2_dims, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(dense_critic_1)
        probs = Dense(self.n_actions, activation='softmax')(dense_actor_2)
        values = Dense(1, activation='linear')(dense_critic_2)

        actor = Model(input=[input, entropia, advantage], output=[probs])
        critic = Model(input=[input], output=[values])
        actor.summary()
        policy = Model(input=[input], output=[probs])

        if (env_name != ''):
            actor.load_weights(env_name + '_actor.h5')        
            critic.load_weights(env_name + '_critic.h5')
            with open (env_name + '_scores.dat', 'rb') as fp:
                self.score_history = pickle.load(fp)
            
        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out) + self.entropy_coefficient * entropia
            return K.sum(-log_lik*advantage)
    
        actor.compile(optimizer=Adam(lr=self.actor_lr), loss=custom_loss)
        critic.compile(optimizer=Adam(lr=self.critic_lr), loss='mean_squared_error')

        return actor, critic, policy

    def choose_action(self, stacked_observation):
        state = stacked_observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0] 
        action = np.random.choice(self.action_space, p=probabilities)

        # print(probabilities)
        #Il clipping delle probabilità evita il logaritmo -inf
        self.entropy = - (tf.math.reduce_sum(probabilities * tf.math.log(tf.clip_by_value(probabilities,1e-8,1.0 - 1e-8))))
        return action

    def learn(self, current_stacked_state, action, reward, state_, done):
        state = current_stacked_state[np.newaxis,:]

        # s_, dump = self.stack_frames(state_)
        state_ = state_[np.newaxis,:]
        # state_ = state_[np.newaxis,:]


        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(state_)[0]
        # print('Val: %2.3f', value)
        # print('NextVal: %2.3f', next_value)

        if done:
            target = reward + self.discount_factor * next_value * 0
        else:
            target = reward + self.discount_factor * next_value
        
        advantage = target - value
        # print(advantage)
        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1
        # print(target.shape)
        # target = np.reshape(target, (1, target.shape[1]))


        self.actor.fit([state, np.reshape(self.entropy, (1, 1)), advantage], actions, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

    def save(self, envName):
        self.actor.save_weights(envName + '_actor.h5')        
        self.critic.save_weights(envName + '_critic.h5')
        with open(envName + '_scores.dat', 'wb') as fp:
            pickle.dump(self.score_history, fp)

        