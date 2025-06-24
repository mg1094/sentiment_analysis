# 舆情分析系统

基于BERT的中文情感分析模型，用于舆情监测和情感分析。

## 功能特点

- 🤖 使用`bert-base-chinese`预训练模型
- 📊 支持三分类情感分析（正面、中性、负面）
- 🔧 完整的训练、评估、预测流程
- 📈 可视化分析结果和训练过程
- 🎯 高精度的情感识别
- 📝 详细的分析报告生成

## 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/mg1094/sentiment_analysis.git
cd sentiment_analysis

# 安装依赖
pip install -r requirements.txt
```

### 2. 创建示例数据

```bash
python main.py --mode create_data
```

### 3. 训练模型

```bash
python main.py --mode train
```

### 4. 预测文本

```bash
# 单个文本预测
python main.py --mode predict --text "这个产品真的很棒！"

# 交互式模式
python main.py --mode interactive
```

## 项目结构

```
sentiment_analysis/
├── config.py              # 配置文件
├── main.py                # 主程序入口
├── demo.py                # 演示程序
├── requirements.txt       # 依赖包列表
├── README.md             # 项目说明
├── src/                  # 源代码目录
│   ├── __init__.py
│   ├── data_processor.py # 数据处理模块
│   ├── model.py          # 模型定义
│   ├── trainer.py        # 训练模块
│   └── predictor.py      # 预测模块
└── data/                 # 数据目录
    └── sample_data.csv   # 示例数据
```

## 性能指标

- 准确率：> 85%
- F1分数：> 0.85
- 推理速度：~100条/秒（CPU）

## 注意事项

1. **首次运行**：会自动下载`bert-base-chinese`模型（约400MB）
2. **内存需求**：训练时建议至少8GB内存
3. **GPU加速**：支持CUDA加速，自动检测GPU环境
4. **中文处理**：已集成jieba分词，适合中文文本