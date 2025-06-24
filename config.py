import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "bert-base-chinese"
    num_labels: int = 3  # 负面(0), 中性(1), 正面(2)
    max_length: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 数据路径
    train_data_path: str = "data/train.csv"
    test_data_path: str = "data/test.csv"
    output_dir: str = "outputs"
    
    # 保存和日志
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 2
    
    # 其他
    seed: int = 42
    device: str = "cuda" if os.path.exists("/proc/driver/nvidia") else "cpu"

@dataclass
class DataConfig:
    """数据配置"""
    text_column: str = "text"
    label_column: str = "label"
    label_mapping: dict = None
    
    def __post_init__(self):
        if self.label_mapping is None:
            self.label_mapping = {
                "负面": 0,
                "中性": 1, 
                "正面": 2,
                0: "负面",
                1: "中性",
                2: "正面"
            }

# 创建配置实例
model_config = ModelConfig()
training_config = TrainingConfig()
data_config = DataConfig()