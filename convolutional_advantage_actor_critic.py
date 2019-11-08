from keras import backend as K
from keras.layers import Dense, Activation, Input, Conv2D, Flatten, Lambda
from keras.models import Model, load_model
from keras.initializers import glorot_normal
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
from keras.utils import plot_model



class Agent(object):
    def __init__(self, actor_lr, critic_lr, stack_size = 1, discount_factor=0.99, n_actions=4,
                 layer1_size=1024, layer2_size=512, input_dims=8, entropy_coefficient = 0.01, state = None, env_name = ''):
        
        self.memory = []
        self.state = state
        self.actor_losses = []
        self.critic_losses = []
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

    # Salvo in memoria l'item (step, stato, nuovo stato, reward, done)
    def remember(self, item):
        self.memory.append(item)

    # Fornisce l'entropia, le probabilità vengono clippate perché è un logaritmo e il risultato può essere NaN
    def entropy(self, probabilities):
        dist = tfp.distributions.Categorical(probs=probabilities)
        probabilities = tf.clip_by_value(probabilities,1e-5,1.0 - 1e-5)
        entropy = dist.entropy()
        return entropy[0]

    # Restituisce un sample dalla distribuzione categorica
    def action(self, probabilities):
        dist = tfp.distributions.Categorical(probs=probabilities)
        action = dist.sample()
        return action

    # Logaritmo della probabilità dell'azione
    def logp(self, args):
        probabilities, action = args
        probabilities = tf.clip_by_value(probabilities,1e-5,1.0 - 1e-5)
        dist = tfp.distributions.Categorical(probs = probabilities)
        logp = dist.log_prob(action)
        return logp


    def choose_action(self, state):
        action = self.actor.predict(state)
        return action[0]

    def get_value(self, state):
        value = self.critic.predict(state)
        return value[0]

    # Entropia della distribuzione delle azioni
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
        input = Input(shape=(93, 84, self.stack_size))
        head =  Conv2D(32, kernel_size=(8, 8), strides=2, activation='relu')(input)
        conv1 = Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu')(head)
        conv2 = Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu')(conv1)
        input_tail_network = Flatten()(conv2)
        
        ### ACTOR ###
        dense_actor_1 =     Dense(self.fc1_dims, activation='relu', kernel_initializer=glorot_normal())(input_tail_network)
        dense_actor_2 =    Dense(self.fc2_dims, activation='relu', kernel_initializer=glorot_normal())(dense_actor_1)
        probs = Dense(self.n_actions, activation='softmax')(dense_actor_2)
        action_layer = Lambda(self.action, output_shape=(1,), name='action')(probs)
        self.actor = Model(input, action_layer)
        plot_model(self.actor, to_file='actor.png', show_shapes=True)

        logp = Lambda(self.logp,
                output_shape=(1,),
                name='logp')([probs, action_layer])

        self.logp_model = Model(input, logp)
        plot_model(self.logp_model, to_file='logp.png', show_shapes=True)

        ### ENTROPIA ###
        entropy = Lambda(self.entropy,
                    output_shape=(1,),
                    name='entropy')(probs)
        self.entropy_model = Model(input, entropy)
        plot_model(self.entropy_model, to_file='entropy_model.png', show_shapes=True)

        loss = self.custom_loss(self.get_entropy(self.state), self.entropy_coefficient)
        self.logp_model.compile(optimizer=RMSprop(lr=self.actor_lr), loss=loss)

        ### CRITIC ###
        dense_critic_1 =    Dense(self.fc1_dims, activation='relu', kernel_initializer=glorot_normal())(input_tail_network)
        dense_critic_2 =    Dense(self.fc2_dims, activation='relu', kernel_initializer=glorot_normal())(dense_critic_1)
        values = Dense(1, activation='linear', kernel_initializer='zero')(dense_critic_2)
        self.critic = Model(input, values)
        self.critic.compile(optimizer=Adam(lr=self.critic_lr), loss='mean_squared_error')
        plot_model(self.critic, to_file='critic.png', show_shapes=True)

        if (env_name != ''):
            self.logp_model.load_weights(env_name + '_actor.h5')        
            self.critic.load_weights(env_name + '_critic.h5')
            with open (env_name + '_scores.dat', 'rb') as fp:
                self.score_history = pickle.load(fp)
            with open(env_name + '_actor_losses.dat', 'rb') as fp:
                self.actor_losses = pickle.load(fp)
            with open(env_name + '_critic_losses.dat', 'rb') as fp:
                self.critic_losses = pickle.load(fp) 


        return 


    def train(self, item, gamma=1.0):
        [step, state, next_state, discounted_reward, done] = item

        # Salvo lo stato per calcolare l'entropia
        self.state = state

        discount_factor = gamma**step


        # a2c: delta = discounted_reward - value
        val = self.get_value(state)[0]
        delta = discounted_reward - val
        
        # Applico il fattore di discount
        discounted_delta = delta * discount_factor
        discounted_delta = np.reshape(discounted_delta, [-1, 1])
        # verbose = 1 if done else 0
        verbose = 0


        # Addestrare il logp_model implica addestrare anche il modello con le probabilità
        logp_history = self.logp_model.fit(state,
                            discounted_delta,
                            batch_size=1,
                            epochs=1,
                            verbose=verbose)
        
        if done:
            self.actor_losses.append(logp_history.history['loss'])

        # in A2C, target = (discounted_reward)
        discounted_delta = discounted_reward
        discounted_delta = np.reshape(discounted_delta, [-1, 1])
        
        critic_history = self.critic.fit(state,
                                discounted_delta,
                                batch_size=1,
                                epochs=1,
                                verbose=verbose)
        if done:
            self.critic_losses.append(critic_history.history['loss'])


    def train_by_episode(self, last_value=0):
        gamma = self.discount_factor
        r = last_value

        # La memoria viene elaborata dalla fine
        for item in self.memory[::-1]:
            [step, state, next_state, reward, done] = item
            # calcola il ritorno (discounted reward)
            r = reward + gamma*r
            item = [step, state, next_state, r, done]
            self.train(item, self.discount_factor)

        return


    def save(self, envName):
        self.logp_model.save_weights(envName + '_actor.h5')        
        self.critic.save_weights(envName + '_critic.h5')
        with open(envName + '_scores.dat', 'wb') as fp:
            pickle.dump(self.score_history, fp)
        with open(envName + '_actor_losses.dat', 'wb') as fp:
            pickle.dump(self.actor_losses, fp)
        with open(envName + '_critic_losses.dat', 'wb') as fp:
            pickle.dump(self.critic_losses, fp)

