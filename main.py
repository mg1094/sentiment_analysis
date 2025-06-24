#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舆情分析主程序
使用BERT模型进行中文情感分析
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_processor import DataProcessor, analyze_data_distribution
from src.model import SentimentClassifier, ModelManager
from src.trainer import Trainer, main as train_main
from src.predictor import SentimentPredictor, SentimentAnalyzer
from config import training_config, model_config, data_config

def create_sample_data():
    """创建示例数据"""
    processor = DataProcessor()
    sample_file = processor.create_sample_data("data/train.csv")
    print(f"示例数据已创建: {sample_file}")
    return sample_file

def train_model(data_path=None):
    """训练模型"""
    print("=== 开始训练舆情分析模型 ===")
    
    if data_path:
        training_config.train_data_path = data_path
    
    # 运行训练
    train_main()

def predict_text(model_path, text=None, file_path=None, output_path=None):
    """预测文本情感"""
    predictor = SentimentPredictor(model_path)
    
    if text:
        # 单个文本预测
        result = predictor.predict_single(text)
        print("=== 预测结果 ===")
        print(f"文本: {result['text']}")
        print(f"情感: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"概率分布: {result['probabilities']}")
        
    elif file_path:
        # 文件批量预测
        analyzer = SentimentAnalyzer(predictor)
        analysis = analyzer.analyze_text_file(file_path)
        
        if analysis:
            print("=== 分析结果 ===")
            print(f"总文本数: {analysis['total_texts']}")
            print("情感分布:")
            for sentiment, count in analysis['sentiment_counts'].items():
                ratio = analysis['sentiment_ratios'][sentiment]
                print(f"  {sentiment}: {count} ({ratio:.1%})")
            
            # 保存结果
            if output_path:
                predictor.save_results_to_csv(analysis['results'], output_path)
                analyzer.generate_sentiment_report(analysis, output_path.replace('.csv', '_report.txt'))
                predictor.plot_sentiment_distribution(analysis, output_path.replace('.csv', '_distribution.png'))
    else:
        print("请提供要预测的文本或文件路径")

def interactive_mode():
    """交互式模式"""
    print("=== 舆情分析交互模式 ===")
    print("输入 'quit' 退出程序")
    
    # 检查是否有训练好的模型
    model_path = "outputs/best_model.pth"
    if not os.path.exists(model_path):
        model_path = "outputs/final_model.pth"
    
    if not os.path.exists(model_path):
        print("未找到训练好的模型，请先训练模型")
        return
    
    predictor = SentimentPredictor(model_path)
    
    while True:
        text = input("\n请输入要分析的文本: ").strip()
        
        if text.lower() == 'quit':
            print("程序退出")
            break
        
        if not text:
            continue
        
        try:
            result = predictor.predict_single(text)
            print(f"情感: {result['predicted_label']} (置信度: {result['confidence']:.3f})")
            
            # 显示详细概率
            probs = result['probabilities']
            print("详细概率:")
            for sentiment, prob in probs.items():
                print(f"  {sentiment}: {prob:.3f}")
                
        except Exception as e:
            print(f"预测出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="舆情分析工具")
    parser.add_argument('--mode', choices=['train', 'predict', 'interactive', 'create_data'], 
                       default='interactive', help='运行模式')
    parser.add_argument('--data', help='训练数据路径')
    parser.add_argument('--model', help='模型路径')
    parser.add_argument('--text', help='要预测的文本')
    parser.add_argument('--file', help='要预测的文件路径')
    parser.add_argument('--output', help='输出路径')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    if args.mode == 'create_data':
        create_sample_data()
        
    elif args.mode == 'train':
        train_model(args.data)
        
    elif args.mode == 'predict':
        model_path = args.model or "outputs/best_model.pth"
        if not os.path.exists(model_path):
            model_path = "outputs/final_model.pth"
        
        if not os.path.exists(model_path):
            print("未找到模型文件，请先训练模型")
            return
        
        predict_text(model_path, args.text, args.file, args.output)
        
    elif args.mode == 'interactive':
        interactive_mode()

if __name__ == "__main__":
    main()