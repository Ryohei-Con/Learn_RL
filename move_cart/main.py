import argparse
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class QNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.linear3 = nn.Linear(hid_dim, out_dim)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """"
        (batch_size, in_dim) -> (batch_size, out_dim)
        """
        data = self.linear1(data)
        data = F.relu(data)
        data = self.linear2(data)
        data = F.relu(data)
        data = self.linear3(data)
        return data


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, reward, action, next_state, done):
        data = (state, reward, action, next_state, done)
        self.buffer.append(data)

    def get_batch(self):
        sample_data = random.sample(self.buffer, self.batch_size)
        states = []
        rewards = []
        actions = []
        next_states = []
        dones = []
        for data in sample_data:
            states.append(data[0])
            rewards.append(data[1])
            actions.append(data[2])
            next_states.append(data[3])
            dones.append(int(data[4]))
        return torch.Tensor(np.array(states)), torch.Tensor(np.array(rewards)), torch.Tensor(np.array(actions)), torch.Tensor(np.array(next_states)), torch.Tensor(np.array(dones))


class DQNAgent:
    def __init__(self, num_states, num_actions, buffer_size, batch_size, learning_rate, num_episodes):
        self.EPS = 0.5
        self.EPS_diff = (self.EPS - 0.01) / num_episodes
        self.GAMMA = 0.99
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, batch_size)

        self.qnet = QNet(num_states, 128, num_actions)
        self.target_qnet = QNet(num_states, 128, num_actions)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), learning_rate)

    def sync_target(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        return
    
    def lower_eps(self):
        self.EPS -= self.EPS_diff
        return
    
    def get_action(self, state):
        state = torch.Tensor(state).unsqueeze(dim=0)
        if np.random.rand() < self.EPS:
            return np.random.choice(self.num_actions)
        else:
            actions: torch.Tensor = self.qnet(state)
            return actions.data.argmax(dim=1)


    def update(self, state, reward, action, next_state, done):
        self.buffer.add(state, reward, action, next_state, done)
        if len(self.buffer) < self.batch_size:
            return
        states, rewards, actions, next_states, dones = self.buffer.get_batch()
        next_q_val = self.target_qnet(next_states).data.max(dim=1).values
        target = rewards + (1 - dones) * self.GAMMA * next_q_val
        preds = torch.gather(self.qnet(states), 1, actions.unsqueeze(dim=1).to(torch.int64))
        loss = self.loss(preds.squeeze(), target)

        # パラメータの更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_position = -2.4

    def step(self, action):
        # 右に進んでいるときに報酬を与える or 一方通行的な報酬を与える
        next_state, reward, terminated, truncated, info = self.env.step(action)
        reward -= abs(next_state[2])
        # if abs(next_state[1]) < 10:
            # TODO: 速さの単位とかがわからない
        if 1.7 < next_state[0] < 2.2:
            reward += 20
        if self.max_pos < next_state[0] < 2.2:
            # 一方通行的な報酬
            reward += 10
            self.max_pos = next_state[0]

        return next_state, reward, terminated, truncated, info

    def reset(self):
        return np.array([-2.0, 0.0, 0.0, 0.0])


def main():
    parser = argparse.ArgumentParser(description="This executes DQN learning on CartPole")
    parser.add_argument("model_name", help="specify model name")
    args = parser.parse_args()

    model_name = args.model_name
    num_iters = 3
    num_episodes = 10000
    state_dim = 4
    action_space = 2
    buffer_size = 4096
    batch_size = 32
    learning_rate = 0.01
    all_history = []
    try:
        for iter in range(num_iters):
            env = gym.make("CartPole-v1", render_mode="human")
            custom_env = RewardWrapper(env)
            agent = DQNAgent(state_dim, action_space, buffer_size, batch_size, learning_rate, num_episodes)
            reward_history = []
            with tqdm(range(num_episodes)) as pbar:
                for episode in pbar:
                    state = custom_env.reset()
                    total_reward = 0
                    while True:
                        action = agent.get_action(state)
                        next_state, reward, terminated, truncated, info = env.step(int(action))
                        done = truncated or terminated
                        agent.update(state, reward, int(action), next_state, done)
                        total_reward += reward
                        state = next_state
                        if done:
                            break
                    reward_history.append(total_reward)
                    agent.lower_eps()
                    if episode % 20 == 0:
                        agent.sync_target()
                        pbar.set_description(f"iter: {iter}, reward: {sum(reward_history[-20:]) / 20}")
            all_history.append(reward_history)

    finally:
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_weight_file = (output_dir / model_name).with_suffix(".pt")
        model_result_text = (output_dir / model_name).with_suffix(".txt")
        model_result_fig = (output_dir / model_name).with_suffix(".png")

        all_history = np.array(all_history)

        # モデルのrewardを保存
        with open(model_result_text, "w") as f:
            for reward in reward_history:
                f.write(str(reward))
                f.write("\n")
        
        # モデルの重みを保存
        torch.save(agent.qnet.state_dict(), model_weight_file)

        # モデルのreward推移を可視化
        plt.plot(all_history.mean(axis=1))
        plt.title("Total Rewards")
        plt.xlabel("iteration")
        plt.ylabel("reward")
        plt.savefig(model_result_fig)
        plt.show()


if __name__ == "__main__":
    main()
