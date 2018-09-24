# PyTorch APIs
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

from model import QNetwork
from replay_memory import ReplayMemory

BUFFER_SIZE  = int(1e5) # replay buffer size
BATCH_SIZE   = 64       # minibatch size
GAMMA        = 0.99     # discount factor
TAU          = 1e-3     # for soft update of target parameters
LR           = 5e-4     # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(object):
    def __init__(self, state_size, action_size, seed, **kwargs):
        
        # Setup default parameters for learning
        self.batch_size   = kwargs['batch_size']  if 'batch_size' in kwargs else BATCH_SIZE
        self.gamma        = kwargs['gamma'] if 'gamma' in kwargs else GAMMA
        self.tau          = kwargs['tau'] if 'tau' in kwargs else TAU
        self.update_every = kwargs['update_every'] if 'update_every' in kwargs else UPDATE_EVERY
        
        buffer_size  = kwargs['buffer_size'] if 'buffer_size' in kwargs else BUFFER_SIZE
        lr           = kwargs['lr'] if 'lr' in kwargs else LR
        
        self.state_size  = state_size
        self.action_size = action_size
        self.seed        = random.seed(seed)
        
        # Create model and target Q-Network
        self.Q         = QNetwork(action_size, state_size, seed).to(device)
        self.Q_target  = QNetwork(action_size, state_size, seed).to(device)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)
        
        # Setup replay memory
        self.memory = ReplayMemory(buffer_size, self.batch_size, seed)
        
        # Initialize tim step to track updates
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Add new experience
        self.memory.append(state, action, reward, next_state, done)
        
        # update time step, and see if we are 
        # ready to we ahev enough samples
        self.t_step = (self.t_step + 1) % self.update_every
        
        if self.t_step == 0:
            # Check if we have enough sample
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample() # sample from memory
                self.learn(experiences)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.Q.eval()
        with torch.no_grad():
            action_values = self.Q(state)
        self.Q.train()
        
        # Select action using Epsilon-greedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        # Double DQN
        # Action with maximum Q value
        q_action = self.Q(next_states).detach().max(1)[1]
        
        # Q values for the next state state
        Q_target_values = self.Q_target(next_states).detach()
        
        # Get the maximum predicted Q value for the next state
        Q_targets_next = Q_target_values.gather(1, q_action.unsqueeze(1))
        # Calculate the Q target for the current state. The done flag should force
        # the second term to zero for the terminal state.
        Q_targets      = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get the expected Q values using model Q-Network, so we can compute the loss
        Q_expected = self.Q(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss and propgate the gradient to
        # update the network weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        # Update the target Q-Network
        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)