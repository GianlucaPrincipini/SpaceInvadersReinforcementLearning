from keras import backend as K
from keras.layers import Dense, Activation, Input, Conv2D, Flatten
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import pickle
import numpy as np
from math import log
import cv2

stack_size = 4

class Agent(object):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=4,
                 layer1_size=64, layer2_size=64, input_dims=8, is_ram = True, env_name = ''):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.is_ram = is_ram
        self.score_history = []
        self.actor, self.critic, self.policy = self.build_actor_critic_network(env_name)
        self.action_space = [i for i in range(n_actions)]

    # def stack_frames(stacked_frames, state, is_newEpisonde):
    #     stack_size = 4
    #     frame = preprocess(state)
    #     if is_newEpisonde:
    #         #Clear our stack
    #         stacked_frames = deque([np.zeros((110, 84), dtype= np.int) for i in range(stack_size)], maxlen=4)

    #         #since qu're in a new episode copy the same frame 4x
    #         stacked_frames.append(frame)
    #         stacked_frames.append(frame)
    #         stacked_frames.append(frame)
    #         stacked_frames.append(frame)

    #         stacked_state = np.stack(stacked_frames, axis=2)
    #     else:
    #         stacked_frames.append(frame)
    #         stacked_state = np.stack(stacked_frames, axis=2)
    #     return stacked_state, stacked_frames


    def preprocess(self, observation):
        retObs = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        retObs = retObs[26:110,:]
        ret, retObs = cv2.threshold(retObs,1,255,cv2.THRESH_BINARY)
        return np.reshape(retObs,(84,84,1))

    def build_actor_critic_network(self, env_name):
        input = Input(shape=(84, 84, 1))
        delta = Input(shape=[1])
        input_tail_network = input
        if (not self.is_ram):
            head = Conv2D(64, kernel_size=3, activation='relu')(input)
            conv1 = Conv2D(64, kernel_size=3, activation='relu')(head)
            input_tail_network = Flatten()(conv1)
        dense1 = Dense(self.fc1_dims, activation='relu')(input_tail_network)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        actor = Model(input=[input, delta], output=[probs])
        critic = Model(input=[input], output=[values])
        critic.summary()
        policy = Model(input=[input], output=[probs])
        if (env_name != ''):
            actor.load_weights(env_name + '_actor.h5')        
            critic.load_weights(env_name + '_critic.h5')
            with open (env_name + '_scores.dat', 'rb') as fp:
                self.score_history = pickle.load(fp)
            
        self.entropy = 0

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-5, 1-1e-5)
            log_lik = y_true*K.log(out)
            loss = K.sum(-log_lik*delta + 0.01 * self.entropy)
            return loss

        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        return actor, critic, policy

    def choose_action(self, observation):
        state = self.preprocess(observation)[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        #Il clipping delle probabilità evita il logaritmo -inf
        self.entropy = - (tf.math.reduce_sum(probabilities * tf.math.log(tf.clip_by_value(probabilities,1e-5,1.0 - 1e-5))))
        return action

    def learn(self, state, action, reward, state_, done):
        state = self.preprocess(state)[np.newaxis,:]
        state_ = self.preprocess(state_)[np.newaxis,:]
        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta =  target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        # print(actions)
        self.actor.fit([state, delta], actions, verbose=0)

        self.critic.fit(state, target, verbose=0)

    def save(self, envName):
        self.actor.save_weights(envName + '_actor.h5')        
        self.critic.save_weights(envName + '_critic.h5')
        with open(envName + '_scores.dat', 'wb') as fp:
            pickle.dump(self.score_history, fp)

        