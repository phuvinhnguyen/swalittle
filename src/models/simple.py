import torch
import torch.nn as nn
from typing import List

class SimplePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x) -> List[torch.Tensor]: # return loss of each trajectory's transition
        losses = []
        for trajectory in x:
            loss = -torch.log(self.model(torch.stack(trajectory.states))) # (batch, action_dim)
            loss = loss[torch.arange(len(trajectory.states)), trajectory.actions] # (batch, )
            losses.append(loss)
        return losses

    def act(self, state): # generate action with some randomness of the model
        with torch.no_grad():
            probs = self.model(torch.FloatTensor(state))
            action = torch.multinomial(probs, 1).item()
        return action

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))