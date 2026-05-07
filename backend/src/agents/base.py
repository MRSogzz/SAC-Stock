"""
Agent 基類（Abstract Class）
所有 RL Agent 都繼承這個基類，方便之後替換成 TD3、PPO 等。
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """根據觀測選擇動作"""

    @abstractmethod
    def update(self):
        """從 ReplayBuffer 更新網路"""

    @abstractmethod
    def save(self, path: str):
        """儲存模型"""

    @abstractmethod
    def load(self, path: str):
        """載入模型"""