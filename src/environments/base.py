from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, Union
import numpy as np

class Environment(ABC):
    """Base environment interface for reinforcement learning"""
    
    def __init__(self):
        self.action_space = None
        self.observation_space = None
        self.current_step = 0
        self.max_steps = None
    
    @abstractmethod
    def reset(self) -> str:
        """Reset environment to initial state
        
        Returns:
            str: Initial state message for LLM
        """
        raise NotImplementedError
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action in environment
        
        Args:
            action: Action to take
        Returns:
            Tuple containing:
                - next_state (str): Next state message for LLM
                - reward (float): Reward received
                - done (bool): Whether episode is finished
                - info (Dict): Additional information
        """
        raise NotImplementedError
    
    def render(self) -> Optional[str]:
        """Render current state (optional)
        
        Returns:
            Optional[str]: Rendered state or None
        """
        return None
    
    def close(self) -> None:
        """Clean up environment resources"""
        pass
    
    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for reproducibility
        
        Args:
            seed: Random seed
        """
        pass
    
    def get_action_space(self):
        """Get action space specification"""
        return self.action_space
    
    def get_observation_space(self):
        """Get observation space specification"""
        return self.observation_space
