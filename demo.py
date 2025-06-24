#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舆情分析演示脚本
展示完整的训练和预测流程
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def demo_complete_workflow():
    """演示完整工作流程"""
    print("🚀 开始舆情分析系统演示")
    print("=" * 50)
    
    # 1. 创建示例数据
    print("\n📝 步骤1: 创建示例数据")
    from src.data_processor import DataProcessor
    processor = DataProcessor()
    sample_file = processor.create_sample_data()
    print(f"✅ 示例数据已创建: {sample_file}")
    
    # 2. 数据分析
    print("\n📊 步骤2: 数据分析")
    from src.data_processor import analyze_data_distribution
    analyze_data_distribution(sample_file)
    
    # 3. 模型训练（快速演示，使用较少epoch）
    print("\n🤖 步骤3: 模型训练")
    from config import training_config
    from src.trainer import main as train_main
    
    # 临时修改配置以加快演示速度
    original_epochs = training_config.num_epochs
    original_batch_size = training_config.batch_size
    
    training_config.num_epochs = 1  # 演示用，只训练1个epoch
    training_config.batch_size = 8  # 减少内存使用
    
    try:
        train_main()
        print("✅ 模型训练完成")
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("这可能是因为内存不足或缺少GPU，请参考README进行故障排除")
    finally:
        # 恢复原始配置
        training_config.num_epochs = original_epochs
        training_config.batch_size = original_batch_size
    
    # 4. 模型预测演示
    print("\n🔮 步骤4: 模型预测演示")
    demo_prediction()
    
    print("\n🎉 演示完成！")
    print("更多使用方法请查看README.md")

def demo_prediction():
    """演示预测功能"""
    from src.predictor import SentimentPredictor, SentimentAnalyzer
    
    # 查找可用的模型
    model_path = None
    for path in ["outputs/best_model.pth", "outputs/final_model.pth"]:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ 未找到训练好的模型，请先完成训练步骤")
        return
    
    try:
        # 加载模型
        predictor = SentimentPredictor(model_path)
        
        # 测试文本
        test_texts = [
            "这个产品真的很棒，质量超出预期！",
            "客服态度很差，完全不推荐购买。",
            "价格还算合理，功能基本够用。",
            "物流速度很快，包装也很精美！",
            "用了一段时间，感觉一般般。"
        ]
        
        print("测试文本情感分析结果:")
        print("-" * 40)
        
        for i, text in enumerate(test_texts, 1):
            result = predictor.predict_single(text)
            sentiment = result['predicted_label']
            confidence = result['confidence']
            
            # 根据情感添加表情符号
            emoji_map = {'正面': '😊', '负面': '😞', '中性': '😐'}
            emoji = emoji_map.get(sentiment, '🤔')
            
            print(f"{i}. {text}")
            print(f"   情感: {sentiment} {emoji} (置信度: {confidence:.3f})")
            print()
        
        # 批量分析
        print("📈 批量分析结果:")
        analysis = predictor.analyze_sentiment_distribution(test_texts)
        
        print(f"总文本数: {analysis['total_texts']}")
        print("情感分布:")
        for sentiment, count in analysis['sentiment_counts'].items():
            ratio = analysis['sentiment_ratios'][sentiment]
            print(f"  {sentiment}: {count} 条 ({ratio:.1%})")
        
    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")

def demo_interactive():
    """交互式演示"""
    print("\n🎮 交互式情感分析")
    print("输入文本来分析情感，输入 'quit' 退出")
    
    from src.predictor import SentimentPredictor
    
    # 查找模型
    model_path = None
    for path in ["outputs/best_model.pth", "outputs/final_model.pth"]:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ 未找到模型，请先运行完整演示")
        return
    
    predictor = SentimentPredictor(model_path)
    
    while True:
        try:
            text = input("\n请输入文本: ").strip()
            
            if text.lower() == 'quit':
                print("👋 再见！")
                break
            
            if not text:
                continue
            
            result = predictor.predict_single(text)
            sentiment = result['predicted_label']
            confidence = result['confidence']
            
            emoji_map = {'正面': '😊', '负面': '😞', '中性': '😐'}
            emoji = emoji_map.get(sentiment, '🤔')
            
            print(f"情感: {sentiment} {emoji}")
            print(f"置信度: {confidence:.3f}")
            
            # 显示概率分布
            probs = result['probabilities']
            print("概率分布:")
            for sent, prob in probs.items():
                bar_length = int(prob * 20)  # 创建简单的进度条
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {sent}: {bar} {prob:.3f}")
                
        except KeyboardInterrupt:
            print("\n👋 程序被用户中断")
            break
        except Exception as e:
            print(f"❌ 出现错误: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        demo_interactive()
    else:
        demo_complete_workflow()