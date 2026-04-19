import torch
import numpy as np
import matplotlib.pyplot as plt

def save_result_text(all_history: np.ndarray, model_result_text: list):
    with open(model_result_text, "w") as f:
        for i, reward_history in enumerate(all_history):
            f.write(f"iter {i}\n")
            for reward in reward_history:
                f.write(f"{reward}\n")


def save_weights(agent, model_weight_file):
    torch.save(agent.qnet.state_dict(), model_weight_file)

def save_figure(all_history: np.ndarray, model_result_fig):
    plt.plot(all_history.mean(axis=0))
    plt.title("Total Rewards")
    plt.xlabel("iteration")
    plt.ylabel("reward")
    plt.savefig(model_result_fig)