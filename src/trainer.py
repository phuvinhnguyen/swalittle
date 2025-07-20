import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
import argparse
import yaml
import importlib
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import time
import json

# ---------- Logger ----------
class Logger:
    def __init__(self):
        self.losses = []
        self.rewards = []

    def log(self, loss, reward):
        self.losses.append(loss)
        self.rewards.append(reward)

    def save_logs(self, name):
        np.savez(f"{name}_logs.npz", loss=self.losses, reward=self.rewards)

    def plot(self, path):
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.losses, label="Loss")
        plt.title(f"{os.path.basename(path)} - Loss")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.rewards, label="Reward")
        plt.title(f"{os.path.basename(path)} - Reward")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{path}_plot.png")

# ---------- Trajectory ----------
@dataclass
class Trajectory:
    states: list
    actions: list
    rewards: list
    losses: list

# ---------- Dynamic Import Utility ----------
def dynamic_import(module_path, class_name):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

# ---------- Trainer ----------
class Trainer:
    def __init__(self,
                 env,
                 policy,
                 algorithm,
                 log_name="RL",
                 episodes_per_batch=10,
                 report_folder="reports",
                 epochs=100,
                 max_step_each_time=[100],
                 learning_rate=1e-2,
                 max_history_trajectory=None,
                 optimizer_name='Adam',
                 ):
        self.env = env
        self.policy = policy
        self.algorithm = algorithm
        self.optimizer = getattr(torch.optim, optimizer_name)(policy.parameters(), lr=learning_rate)
        self.episodes_per_batch = episodes_per_batch
        self.logger = Logger()
        self.log_name = log_name
        self.report_folder = report_folder
        self.epochs = epochs
        self.max_step_each_time = max_step_each_time
        self.learning_rate = learning_rate
        self.max_history_trajectory=max_history_trajectory if isinstance(max_history_trajectory, int) else episodes_per_batch
        self.trajectories = []
        os.makedirs(report_folder, exist_ok=True)
        self.compute_full_max_step()

    def collect_trajectories(self, max_step):
        trajectories = []
        batch_rewards = []

        for _ in range(self.episodes_per_batch):
            state = self.env.reset()
            states, actions, rewards, losses = [], [], [], []
            done = False

            while (not done) and (len(states) < max_step):
                action = self.policy.act(state)

                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                losses.append(0.)

                state = next_state

            batch_rewards.append(sum(rewards))
            trajectories.append(Trajectory(states, actions, rewards, losses))

        # save trajectories to file
        try:
            file_path = os.path.join(self.report_folder, f'{self.log_name}_trajectories.json')
            file_trajectories = []
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    file_trajectories = json.load(f)

            current_trajectory = [[str(s.states), str(s.actions), str(s.rewards)] for s in trajectories]
            file_trajectories += [{'trajectory': current_trajectory}]

            with open(file_path, 'w') as f:
                json.dump(file_trajectories, f, indent=4)
        except Exception as e:
            print('ERROR: cannot save trajectories - ', e)

        self.trajectories = (self.trajectories + trajectories)[:self.max_history_trajectory]

        return self.trajectories, batch_rewards

    def compute_full_max_step(self):
        if len(self.max_step_each_time) == 1:
            self.max_step_each_time *= self.epochs
            return

        k = len(self.max_step_each_time) - 1
        segment_sizes = [self.epochs // k + (1 if i < self.epochs % k else 0) for i in range(k)]

        full = []
        for i, seg_len in enumerate(segment_sizes):
            start, end = self.max_step_each_time[i], self.max_step_each_time[i + 1]
            full += [int(round(start + (end - start) * j / seg_len)) for j in range(seg_len)]

        self.max_step_each_time = full[:self.epochs]

    def train(self):
        for epoch, max_step in zip(range(self.epochs), self.max_step_each_time):
            start_time = time.time()
            trajectories, rewards = self.collect_trajectories(max_step)
            
            if not getattr(self.algorithm, 'self_compute', False):
                losses = self.policy(trajectories) # loss has same size as trajectories, we need to parse loss to correct transition in the trajectory

                for i, loss in enumerate(losses):
                    trajectories[i].losses = loss

            loss = self.algorithm.compute_loss(trajectories)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_time = time.time() - start_time

            avg_reward = np.mean(rewards)
            self.logger.log(loss.item(), avg_reward)
            print(f"[{self.log_name}] Epoch {epoch:3d} | Time: {running_time:.2f}s | Max Step: {max_step} | Loss: {loss.item():.3f} | Avg Reward: {avg_reward:.2f}")

    def test(self, episodes=5, render=False):
        total = 0
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            reward_sum = 0
            while not done:
                action = self.policy.act(state)
                state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                if render:
                    self.env.env.render()
            total += reward_sum
        print(f"[{self.log_name}] Test Avg Reward over {episodes} episodes: {total / episodes:.2f}")

    def save_model(self):
        # save model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.policy.save(os.path.join(self.report_folder, f'model_{self.log_name}_{timestamp}.pth'))

    def load_model(self, path="model.pth"):
        self.policy.load(path)

    def report(self):
        self.logger.save_logs(os.path.join(self.report_folder, self.log_name))
        self.logger.plot(os.path.join(self.report_folder, self.log_name))

# ---------- Config-driven Main ----------
def main():
    parser = argparse.ArgumentParser(description="RL Trainer with dynamic import")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Dynamic import for environment
    env_cfg = config['environment']
    env_module, env_class = env_cfg['class'].rsplit('.', 1)
    EnvClass = dynamic_import(f"environments.{env_module}", env_class)
    env = EnvClass(**env_cfg.get('params', {}))

    # Dynamic import for model
    model_cfg = config['model']
    model_module, model_class = model_cfg['class'].rsplit('.', 1)
    ModelClass = dynamic_import(f"models.{model_module}", model_class)
    model = ModelClass(**model_cfg.get('params', {}))

    # Dynamic import for algorithm
    algo_cfg = config['algorithm']
    algo_module, algo_class = algo_cfg['class'].rsplit('.', 1)
    AlgoClass = dynamic_import(f"algorithms.{algo_module}", algo_class)
    algorithm = AlgoClass(model, **algo_cfg.get('params', {}))

    # Dynamic import for trainer
    trainer_cfg = config.get('trainer', {})
    trainer = Trainer(env, model, algorithm, **trainer_cfg)

    # Train
    trainer.train()
    trainer.test()
    trainer.report()
    trainer.save_model()

if __name__ == "__main__":
    main()