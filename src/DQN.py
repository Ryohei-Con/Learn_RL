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

from common.save_models import save_result_text, save_weights, save_figure
from common.replay_buffer import ReplayBuffer
from common.QNet import QNet

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
            env = gym.make("CartPole-v1", render_mode=None)
            agent = DQNAgent(state_dim, action_space, buffer_size, batch_size, learning_rate, num_episodes)
            reward_history = []
            with tqdm(range(num_episodes)) as pbar:
                for episode in pbar:
                    state, info = env.reset()
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

        save_result_text(all_history, model_result_text)
        save_weights(agent, model_weight_file)
        save_figure(all_history, model_result_fig)


if __name__ == "__main__":
    main()
