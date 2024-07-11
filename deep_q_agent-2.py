import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from tqdm.notebook import tqdm


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


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.2)
        
        # Output layer
        self.out = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        x = self.out(x)
        return x

class DQLAgent:
    def __init__(self, state_dim, lr=5e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=64, memory_size=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = 3 * 3 * 3 * 3  # 3 options each for bid_price_delta, ask_price_delta, bid_quantity, ask_quantity
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model = QNetwork(state_dim, self.action_dim).to(self.device)
        self.target_model = QNetwork(state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval() 
        with torch.no_grad():
            act_values = self.model(state)
        self.model.train() 
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(-1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(-1).to(self.device) 
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones).astype(np.uint8)).unsqueeze(-1).to(self.device) 
    
        Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(-1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones)) 
    
        Q_expected = self.model(states).gather(1, actions)

        # print("Shapes of tensors:")
        # print("Q_targets_next shape:", Q_targets_next.shape)
        # print("Q_targets shape:", Q_targets.shape)
        # print("Q_expected shape:", Q_expected.shape)
        # print("rewards shape:", rewards.shape)
        # print("dones shape:", dones.shape)
        # print("actions shape:", actions.shape)

        loss = torch.nn.functional.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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


    def get_reward(self, state):
        return state["realized_pnl"]


    def train(self, env, num_episodes=1000, epsilon=1.0):
        total_rewards = []
        for episode in tqdm(range(num_episodes)):
            self.epsilon = epsilon
            raw_state, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                state = extract_features(raw_state)
                action_index = self.act(state)
                decoded_action = self.decode_action(raw_state["bid_ask"], action_index)
                next_raw_state, done = env.step(decoded_action)
                reward = self.get_reward(raw_state)
                next_state = extract_features(next_raw_state)
                
                self.remember(state, action_index, reward, next_state, done)
                raw_state = next_raw_state
                state = next_state
                total_reward += reward
                    
                self.replay()
            total_rewards.append(total_reward)
            print(f"Episode: {episode+1}/{num_episodes}, Total Reward: {total_reward}")
            self.update_target_model()
        return total_rewards

            
    def exploit(self, env, num_episodes=100):
        original_epsilon = self.epsilon 
        self.epsilon = 0 
        total_rewards = []
        for episode in tqdm(range(num_episodes)):
            raw_state, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                state = extract_features(raw_state)
                action_index = self.act(state)
                decoded_action = self.decode_action(raw_state["bid_ask"], action_index)
                next_raw_state, done = env.step(decoded_action)
                reward = self.get_reward(raw_state)
                next_state = extract_features(next_raw_state)
                
                total_reward += reward
            total_rewards.append(total_reward)
            print(f"Exploit Episode: {episode+1}/{num_episodes}, Total Reward: {total_reward}")
        return total_rewards
        