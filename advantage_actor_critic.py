from keras import backend as K
from keras.layers import Dense, Activation, Input, Conv2D, Flatten, Lambda
from keras.models import Model, load_model
from keras.initializers import Orthogonal
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import TimeDistributed
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import pickle
import numpy as np
from math import log
import cv2
from collections import deque
import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, actor_lr, critic_lr, stack_size = 4, discount_factor=0.99, n_actions=4,
                 layer1_size=1024, layer2_size=1024, input_dims=8, entropy_coefficient = 0.01, state = None, env_name = ''):
        
        # s,a,r,s' are stored in memory
        self.memory = []
        self.state = state

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
        self.build_actor_critic_network(env_name)
        self.action_space = [i for i in range(n_actions)]

    def reset_memory(self):
        self.memory = []

    # remember every s,a,r,s' in every step of the episode
    def remember(self, item):
        self.memory.append(item)

    def entropy(self, probabilities):
        dist = tfp.distributions.Categorical(probs=probabilities)
        probabilities = tf.clip_by_value(probabilities,1e-8,1.0 - 1e-8)
        entropy = dist.entropy()
        return entropy[0]

    def action(self, probabilities):
        dist = tfp.distributions.Categorical(probs=probabilities)
        action = dist.sample(1)
        return action

    # given mean, stddev, and action compute
    def logp(self, args):
        probabilities, action = args
        probabilities = tf.clip_by_value(probabilities,1e-8,1.0 - 1e-8)
        dist = tfp.distributions.Categorical(probs = probabilities)
        logp = dist.log_prob(action)
        # tf.print(logp)
        return logp


    def choose_action(self, state):
        action = self.actor.predict(state)
        return action[0]

    def get_value(self, state):
        value = self.critic.predict(state)
        return value[0]

    # return the entropy of the policy distribution
    def get_entropy(self, state):
        entropy = self.entropy_model.predict(state)
        return entropy[0]


    # logp loss, the 3rd and 4th variables (entropy and beta) are needed
    # by A2C so we have a different loss function structure
    def custom_loss(self, entropy, entropy_coefficient=0.0):
        def loss(y_true, y_pred):
            return -K.mean((y_pred * y_true) + (entropy_coefficient * entropy), axis=-1)

        return loss


    def build_actor_critic_network(self, env_name):
        #creazione della struttura delle nostre due reti
        # input = Input(shape=(93, 84, self.stack_size))
        # head =  Conv2D(32, kernel_size=(8, 8), strides=2, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(input)
        # conv1 = Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(head)
        # conv2 = Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(conv1)
        # input_tail_network = Flatten()(conv2)
        input_tail_network = Input(shape=self.input_dims)
        input = input_tail_network
        
        ### ACTOR ###
        dense_actor_1 =     Dense(self.fc1_dims, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(input_tail_network)
        dense_actor_2 =     Dense(self.fc1_dims, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(dense_actor_1)
        probs = Dense(self.n_actions, activation='softmax')(dense_actor_2)
        action_layer = Lambda(self.action, output_shape=(1,), name='action')(probs)
        self.actor = Model(input=[input], output=[action_layer])

        logp = Lambda(self.logp,
                output_shape=(1,),
                name='logp')([probs, action_layer])

        self.logp_model = Model(input=[input], output=[logp])

        ### ENTROPIA ###
        entropy = Lambda(self.entropy,
                    output_shape=(1,),
                    name='entropy')([probs])
        self.entropy_model = Model(input, entropy)

        loss = self.custom_loss(self.get_entropy(self.state), self.entropy_coefficient)
        self.logp_model.compile(optimizer=RMSprop(lr=self.actor_lr), loss=loss)

        ### CRITIC ###
        dense_critic_1 =    Dense(self.fc2_dims, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(input_tail_network)
        dense_critic_2 =    Dense(self.fc2_dims, activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)))(dense_critic_1)
        values = Dense(1, activation='linear')(dense_critic_2)
        self.critic = Model(input=[input], output=[values])
        self.critic.compile(optimizer=Adam(lr=self.critic_lr), loss='mean_squared_error')

        if (env_name != ''):
            self.logp_model.load_weights(env_name + '_actor.h5')        
            self.critic.load_weights(env_name + '_critic.h5')
            with open (env_name + '_scores.dat', 'rb') as fp:
                self.score_history = pickle.load(fp)


        return 


    # main routine for training as used by all 4 policy gradient
    # methods
    def train(self, item, gamma=1.0):
        [step, state, next_state, discounted_reward, done] = item

        # must save state for entropy computation
        self.state = state

        discount_factor = gamma**step


        # a2c: delta = discounted_reward - value
        delta = discounted_reward - self.get_value(state)[0] 

        # apply the discount factor as shown in Algortihms
        # 10.2.1, 10.3.1 and 10.4.1
        discounted_delta = delta * discount_factor
        discounted_delta = np.reshape(discounted_delta, [-1, 1])
        # print(discounted_delta)
        # verbose = 1 if done else 0
        verbose = 0

        # train the logp model (implies training of actor model
        # as well) since they share exactly the same set of
        # parameters
        self.logp_model.fit(np.array(state),
                            discounted_delta,
                            batch_size=1,
                            epochs=1,
                            verbose=verbose)

        # in A2C, the target value is the return (discounted_reward
        # replaced by return in the train_by_episode function)
        discounted_delta = discounted_reward
        discounted_delta = np.reshape(discounted_delta, [-1, 1])

        # train the value network (critic)
        self.critic.fit(np.array(state),
                                discounted_delta,
                                batch_size=1,
                                epochs=1,
                                verbose=verbose)


    # train by episode (REINFORCE, REINFORCE with baseline
    # and A2C use this routine to prepare the dataset before
    # the step by step training)
    def train_by_episode(self, last_value=0):
        # implements A2C training from the last state
        # to the first state
        # discount factor
        gamma = self.discount_factor
        r = last_value
        # the memory is visited in reverse as shown
        # in Algorithm 10.5.1
        mem = list(self.memory)
        for item in mem[::-1]:
            [step, state, next_state, reward, done] = item
            # compute the return
            r = reward + gamma*r
            item = [step, state, next_state, r, done]
            # train per step
            # reward has been discounted
            self.train(item)

        return


    def save(self, envName):
        self.logp_model.save_weights(envName + '_actor.h5')        
        self.critic.save_weights(envName + '_critic.h5')
        with open(envName + '_scores.dat', 'wb') as fp:
            pickle.dump(self.score_history, fp)

        