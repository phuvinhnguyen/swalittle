from abc import ABC, abstractmethod
import torch

class BaseAlgorithm(ABC):
    def __init__(self, policy, gamma=0.99):
        self.policy = policy
        self.gamma = gamma

    @abstractmethod
    def compute_loss(self, trajectories):
        pass

    def compute_returns(self, rewards):
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

# ---------- REINFORCE ----------
class REINFORCE(BaseAlgorithm):
    def compute_loss(self, trajectories):
        loss = 0
        for traj in trajectories:
            returns = self.compute_returns(traj.rewards)
            for _loss, G in zip(traj.losses, returns):
                loss += _loss * G
        return loss / len(trajectories)

# ---------- GRPO ----------
class GRPO(BaseAlgorithm):
    def compute_loss(self, trajectories):
        rewards = torch.tensor([sum(t.rewards) for t in trajectories], dtype=torch.float32)
        z = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)

        loss = 0
        for score, traj in zip(z, trajectories):
            for _loss in traj.losses:
                loss += _loss * score
        return loss / len(trajectories)
    
# ---------- DrGRPO ----------
class DrGRPO(BaseAlgorithm):
    def compute_loss(self, trajectories):
        rewards = torch.tensor([sum(t.rewards) for t in trajectories], dtype=torch.float32)
        z = rewards / (rewards.std(unbiased=False) + 1e-8)

        loss = 0
        for score, traj in zip(z, trajectories):
            for _loss in traj.losses:
                loss += _loss * score
        return loss / len(trajectories)