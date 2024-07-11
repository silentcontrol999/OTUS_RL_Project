import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from collections import deque
from tqdm.notebook import tqdm
import random
import gc


def calculate_returns(data):
    if len(data) < 2:
        return [0]  
    returns = np.diff(data) / data[:-1]
    returns[np.isnan(returns)] = 0
    returns[np.isinf(returns)] = 0
    return returns

def calculate_percentiles(data, percentiles=[1, 5, 10, 25, 50, 90, 95, 99]):
    if len(data) == 0:
        return np.zeros(len(percentiles))
    data_percentiles = np.percentile(data, percentiles)
    return data_percentiles

def extract_features(state):
    bid_prices = [update['bid_price'] for update in state['bid_ask']]
    ask_prices = [update['ask_price'] for update in state['bid_ask']]
    trade_prices = [trade['price'] for trade in state['trades']]
    trade_quantities = [trade['quantity'] for trade in state['trades']]
    
    # price returns
    bid_price_returns = calculate_returns(bid_prices)
    ask_price_returns = calculate_returns(ask_prices)
    trade_price_returns = calculate_returns(trade_prices)
    
    # std of returns
    bid_price_returns_std = np.std(bid_price_returns)
    ask_price_returns_std = np.std(ask_price_returns)
    trade_price_returns_std = np.std(trade_price_returns)
    
    # percentiles for price returns and quantities
    bid_price_returns_percentiles = calculate_percentiles(bid_price_returns)
    ask_price_returns_percentiles = calculate_percentiles(ask_price_returns)
    trade_price_returns_percentiles = calculate_percentiles(trade_price_returns)
    trade_quantities_percentiles = calculate_percentiles(trade_quantities)

    # combine all features
    features = np.concatenate([
        [bid_price_returns_std, ask_price_returns_std, trade_price_returns_std],
        bid_price_returns_percentiles, ask_price_returns_percentiles,
        trade_price_returns_percentiles, trade_quantities_percentiles,
        [state["cash_balance"], state["inventory"], state["realized_pnl"], state["unrealized_pnl"]]
    ])
    
    return features



class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.network(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=2e-4, gamma=0.99, eps_clip=0.2, K_epochs=4, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size

        self.memory = deque(maxlen=10000)

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def remember(self, state, action, prob, reward, next_state, done):
        self.memory.append((state, action, prob, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action).item()

    def decode_action(self, bid_ask, action):
        price_delta_options = [0, 0.1, -0.1]
        quantity_options = [0, 0.005, 0.01]

        # decode the action into the deltas and quantities
        bid_price_delta_index = action % 3
        action //= 3
        ask_price_delta_index = action % 3
        action //= 3
        bid_quantity_index = action % 3
        action //= 3
        ask_quantity_index = action

        bid_price_delta = price_delta_options[bid_price_delta_index]
        ask_price_delta = price_delta_options[ask_price_delta_index]
        bid_quantity = quantity_options[bid_quantity_index]
        ask_quantity = quantity_options[ask_quantity_index]

        # construct the limit orders
        limit_orders = []
        if bid_quantity > 0:
            limit_orders.append({'price': bid_ask[-1]["bid_price"] + bid_price_delta, 'quantity': bid_quantity, 'buySell': 'BUY'})
        if ask_quantity > 0:
            limit_orders.append({'price': bid_ask[-1]["ask_price"] + ask_price_delta, 'quantity': ask_quantity, 'buySell': 'SELL'})

        return limit_orders


    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, old_log_probs, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones).astype(int)).to(self.device)
        
        # Calculate advantages and critic values
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Recalculate log probs and entropy because policy may have changed
            new_log_probs = Categorical(self.actor(states)).log_prob(actions)
        
            # Calculate the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
        
            # Calculate surrogate loss
            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()
        
            # Update actor
            self.optimizer_actor.zero_grad()
            policy_loss.backward()
            self.optimizer_actor.step()
        
            # Calculate value loss
            values_pred = self.critic(states).squeeze()
            value_loss = F.mse_loss(returns.detach(), values_pred)
        
            # Update critic
            self.optimizer_critic.zero_grad()
            value_loss.backward()
            self.optimizer_critic.step()
        
        self.memory.clear()

    def get_reward(self, state):
        return state["realized_pnl"]
    
    def train(self, env, num_episodes=1000):
        total_rewards = []
        for episode in range(num_episodes):
            raw_state, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                state = extract_features(raw_state)
                action_index, log_prob = self.act(state)
                decoded_action = self.decode_action(raw_state["bid_ask"], action_index)  # Assuming env provides current bid_ask
                next_raw_state, done = env.step(decoded_action)
                reward = self.get_reward(raw_state)
                next_state = extract_features(next_raw_state)

                self.remember(state, action_index, log_prob, reward, next_state, done)  # log_prob placeholder with _
                raw_state = next_raw_state
                state = next_state
                total_reward += reward

                if len(self.memory) >= self.batch_size:
                    self.replay()

            total_rewards.append(total_reward)
            print(f"Episode: {episode+1}, Reward: {total_reward}")
        return total_rewards
        

    def exploit(self, env, num_episodes=100):
        original_epsilon = self.epsilon
        self.epsilon = 0  # Set epsilon to 0 to exploit the model
        total_rewards = []

        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            state = extract_features(state)
            total_reward = 0
            done = False
            while not done:
                action, _ = self.act(state)  # Ignore log_prob during exploitation
                next_state, reward, done, _ = env.step(action)
                next_state = extract_features(next_state)
                state = next_state
                total_reward += reward
            total_rewards.append(total_reward)
            print(f"Episode: {episode+1}, Reward: {total_reward}")

        self.epsilon = original_epsilon  # Reset epsilon back to original after exploitation
        return total_rewards




