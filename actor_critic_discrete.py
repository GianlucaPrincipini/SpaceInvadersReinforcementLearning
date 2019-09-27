import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

## RETE NEURALE GENERICA
    # lr            learning rate
    # input_dims    dimensione di input dell'environment
    # fc1_dims      dimensione del primo layer connesso
    # fc2_dims      dimensione del secondo layer
    # n_actions     numero di azioni dell'environment (output)
class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(GenericNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        if (T.cuda.is_available()):
            self.device = T.device("cuda:0")
        else:
            print("Errore, cuda non è disponibile")
    
    def forward(self, observation):
        # Lo stato è un cuda tensor 
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# gamma = discount factor, 
        # Discount factors are associated with time horizons. 
        # Longer time horizons have have much more variance 
        # as they include more irrelevant information, 
        # while short time horizons are biased towards 
        # only short-term gains.

        #. But if your time horizon lasts for the entire month, 
        # then every single thing that makes you feel good 
        # or bad for the entire month will factor 
        # into your judgement
class Agent(object):
    def __init__(self, actor_lr, critic_lr, input_dims, gamma = 0.99,
                l1_size = 256, l2_size = 256, n_actions = 3):
            self.gamma = gamma
            self.log_probs = None
            self.actor = GenericNetwork(actor_lr, input_dims, l1_size, l2_size, n_actions)
            self.critic = GenericNetwork(critic_lr, input_dims, l1_size, l2_size, n_actions)
    
    # Genera la distribuzione di probabilità delle azioni, prende un elemento random
    # dalla distribuzione categorica, ne calcola il logaritmo e lo restituisce
    def choose_action(self, observation):
        probabilities = F.softmax(self.actor.forward(observation))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad() 

        critic_value = self.critic.forward(state)
        critic_value_ = self.critic.forward(new_state)


        delta = ((reward + self.gamma*critic_value_ * (1 - int(done))) - critic_value)

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

