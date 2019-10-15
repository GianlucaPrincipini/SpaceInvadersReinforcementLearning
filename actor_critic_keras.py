from keras import backend as K
from keras.layers import Dense, Activation, Input, Conv2D, Flatten
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np

class Agent(object):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=4,
                 layer1_size=256, layer2_size=256, input_dims=8, is_ram = True):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.is_ram = is_ram

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(n_actions)]



    def build_actor_critic_network(self):
        input = Input(shape=self.input_dims)
        delta = Input(shape=[1])
        input_tail_network = input
        if (not self.is_ram):
            head = Conv2D(32, kernel_size=3, activation='relu')(input)
            conv1 = Conv2D(16, kernel_size=3, activation='relu')(head)
            input_tail_network = Flatten()(conv1)
        dense1 = Dense(self.fc1_dims, activation='relu')(input_tail_network)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-5, 1-1e-5)
            log_lik = y_true*K.log(out)
            loss = K.sum(-log_lik*delta)
            print(loss)
            return loss

        actor = Model(input=[input, delta], output=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)
        critic = Model(input=[input], output=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')
        critic.summary()
        policy = Model(input=[input], output=[probs])

        return actor, critic, policy

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        # print(probabilities)
        return action

    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis,:]
        state_ = state_[np.newaxis,:]
        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta =  target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        # checkpoint
        filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


        # print(actions)
        self.actor.fit([state, delta], actions, verbose=0)

        self.critic.fit(state, target, verbose=1)

