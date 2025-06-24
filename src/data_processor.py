import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Dict, Tuple
import jieba
import re
from config import data_config, model_config

class SentimentDataset(Dataset):
    """情感分析数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 文本预处理
        text = self.preprocess_text(text)
        
        # 编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 去除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(model_config.model_name)
    
    def load_data(self, file_path: str) -> Tuple[List[str], List[int]]:
        """加载数据"""
        try:
            df = pd.read_csv(file_path)
            texts = df[data_config.text_column].tolist()
            
            # 标签处理
            if data_config.label_column in df.columns:
                labels = df[data_config.label_column].tolist()
                # 如果标签是字符串，转换为数字
                if isinstance(labels[0], str):
                    labels = [data_config.label_mapping.get(label, 1) for label in labels]
            else:
                # 如果没有标签列，用于预测
                labels = [1] * len(texts)  # 默认中性
            
            return texts, labels
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return [], []
    
    def create_dataset(self, texts: List[str], labels: List[int]) -> SentimentDataset:
        """创建数据集"""
        return SentimentDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=model_config.max_length
        )
    
    def create_sample_data(self, output_path: str = "data/sample_data.csv"):
        """创建示例数据"""
        import os
        os.makedirs("data", exist_ok=True)
        
        sample_data = {
            "text": [
                "这个产品真的很棒，我非常满意！",
                "服务态度很差，完全不推荐。",
                "还可以吧，没什么特别的感觉。",
                "质量超出预期，值得购买！",
                "价格太贵了，性价比不高。",
                "总体来说还不错，有改进空间。",
                "非常失望，完全不符合描述。",
                "物流很快，包装也很好。",
                "用了几天，感觉一般般。",
                "强烈推荐，真的很好用！"
            ],
            "label": ["正面", "负面", "中性", "正面", "负面", "中性", "负面", "正面", "中性", "正面"]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"示例数据已保存到: {output_path}")
        return output_path

def analyze_data_distribution(file_path: str):
    """分析数据分布"""
    df = pd.read_csv(file_path)
    print("=== 数据分布分析 ===")
    print(f"总样本数: {len(df)}")
    print(f"文本长度统计:")
    text_lengths = df[data_config.text_column].str.len()
    print(f"  平均长度: {text_lengths.mean():.2f}")
    print(f"  最大长度: {text_lengths.max()}")
    print(f"  最小长度: {text_lengths.min()}")
    
    if data_config.label_column in df.columns:
        print("标签分布:")
        label_counts = df[data_config.label_column].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    # 测试数据处理器
    processor = DataProcessor()
    
    # 创建示例数据
    sample_file = processor.create_sample_data()
    
    # 分析数据
    analyze_data_distribution(sample_file)
    
    # 测试数据加载
    texts, labels = processor.load_data(sample_file)
    print(f"\n加载了 {len(texts)} 条数据")
    
    # 创建数据集
    dataset = processor.create_dataset(texts, labels)
    print(f"数据集大小: {len(dataset)}")
    
    # 查看一个样本
    sample = dataset[0]
    print(f"样本形状: input_ids: {sample['input_ids'].shape}, labels: {sample['labels'].shape}")