
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class _DQN_Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(_DQN_Network, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 2)
        self.fc2 = nn.Linear(2, 8)
        self.fc3 = nn.Linear(8, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNLearningAgent:
    def __init__(self, seed,
                 state_size=6,
                 action_size=16,
                 discount_factor=0.995,
                 exploration_rate=1.0,
                 exploration_decay_rate=0.9995):
        
        self.memory = deque(maxlen=2000)
        self.gamma = discount_factor    # discount rate
        self.exploration_rate = exploration_rate
        self.exploration_rate_min = 0.15
        self.exploration_rate_decay = exploration_decay_rate
        self.learning_rate = 0.01 # eta SGD

        self._state_size = state_size 
        self._action_size = action_size
                  
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print('WARNING: No GPU available.  Will continue with CPU.')
        
     
        self.model = _DQN_Network(self._state_size, self._action_size).to(self.device)
        self.target_model = _DQN_Network(self._state_size, self._action_size).to(self.device)
        
        self.update_target_model() 
        self.target_model.eval() 

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss() 
                

    def begin_episode(self, observation):
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
            
        action = random.randrange(self._action_size)
        return action

  
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
      

    def act(self, state):
        if np.random.uniform(0, 1) <= self.exploration_rate:
            return random.randrange(self._action_size)
        
        # state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # [1, state_size]
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        state = state.float().unsqueeze(0).to(self.device)
        
        self.model.eval() 
        with torch.no_grad(): 
            act_values = self.model(state)
        self.model.train() 
            
        return act_values.argmax().item()  


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0, 0 
            
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        Q_current = self.model(states).gather(1, actions)
        with torch.no_grad():
          
            Q_next_max = self.target_model(next_states).max(1)[0].unsqueeze(1)
            Q_target = rewards + (self.gamma * Q_next_max * (1 - dones))

        loss = self.loss_fn(Q_current, Q_target)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        _q = Q_target.mean().item()
        return loss.item(), _q
                
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        return

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.update_target_model()


    def save(self, name):
        torch.save(self.model.state_dict(), name)
