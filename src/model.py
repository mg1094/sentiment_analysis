import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from config import model_config

class SentimentClassifier(nn.Module):
    """基于BERT的情感分类器"""
    
    def __init__(self, model_name: str = None, num_labels: int = None, freeze_bert: bool = False):
        super(SentimentClassifier, self).__init__()
        
        # 使用配置中的参数
        self.model_name = model_name or model_config.model_name
        self.num_labels = num_labels or model_config.num_labels
        
        # 加载BERT模型
        self.bert = BertModel.from_pretrained(
            self.model_name,
            return_dict=True
        )
        
        # 冻结BERT参数（可选）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 分类头
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        
        # 初始化分类器权重
        self._init_weights(self.classifier)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask, labels=None):
        """前向传播"""
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 获取[CLS]标记的输出
        pooled_output = outputs.pooler_output
        
        # Dropout和分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.last_hidden_state
        }
    
    def predict(self, input_ids, attention_mask):
        """预测"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            predictions = torch.nn.functional.softmax(outputs['logits'], dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)
        return predicted_labels, predictions

class ModelManager:
    """模型管理器"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_model(self, num_labels: int = None, freeze_bert: bool = False):
        """创建模型"""
        self.model = SentimentClassifier(
            num_labels=num_labels,
            freeze_bert=freeze_bert
        )
        self.model.to(self.device)
        return self.model
    
    def save_model(self, model, save_path: str):
        """保存模型"""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'model_name': model.model_name,
                'num_labels': model.num_labels
            }
        }, save_path)
        print(f"模型已保存到: {save_path}")
    
    def load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型
        config = checkpoint['model_config']
        self.model = SentimentClassifier(
            model_name=config['model_name'],
            num_labels=config['num_labels']
        )
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型已从 {model_path} 加载")
        return self.model
    
    def get_model_info(self):
        """获取模型信息"""
        if self.model is None:
            return "模型未加载"
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            "模型名称": self.model.model_name,
            "标签数量": self.model.num_labels,
            "总参数数": f"{total_params:,}",
            "可训练参数数": f"{trainable_params:,}",
            "设备": str(self.device)
        }
        
        return info