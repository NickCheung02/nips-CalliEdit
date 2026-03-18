import torch
import torch.nn as nn
import os
import sys
from easydict import EasyDict as edict  # <--- 【新增这一行导入】

# 确保能正确导入 s_ocr_recog
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s_ocr_recog.RecModel import RecModel

class CalligraphyStyleExtractor(nn.Module):
    def __init__(self, weight_path='./s_ocr_weights/ppv3_rec.pth', device='cuda'):
        super().__init__()
        # PP-OCRv3 官方的中英文识别模型配置 (保持与 AnyText 一致)
        # === 【修改】：用 edict() 将普通字典转为可通过 . 访问属性的对象 ===
        config = edict({
            'in_channels': 3,
            'backbone': {'type': 'MobileNetV1Enhance', 'scale': 0.5, 'last_conv_stride': [1, 2], 'last_pool_type': 'avg'},
            'neck': {'type': 'SequenceEncoder', 'encoder_type': 'svtr', 'dims': 64, 'depth': 2, 'hidden_dims': 120, 'use_guide': True},
            'head': {'type': 'CTCHead', 'fc_decay': 1e-05, 'out_channels': 6625, 'return_feats': True}
        })
        self.model = RecModel(config)
        
        # 加载预训练权重
        print(f"Loading Calligraphy Style Extractor weights from {weight_path}...")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found at {weight_path}. Please make sure you copied s_ocr_weights correctly.")
            
        state_dict = torch.load(weight_path, map_location='cpu')
        # 直接使用原生 PyTorch 方式加载
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        # 冻结所有参数，仅作为特征提取器，不参与反向传播
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, ref_imgs):
        """
        输入: ref_imgs [B, 3, 48, W] 范围在 [-1, 1] 的归一化参考图张量
        输出: [B, SeqLen, 512] 的展平风格序列特征
        """
        # 提取 Backbone 的 2D 空间特征图 [B, 512, H', W']
        feat_2d = self.model.extract_style_feature(ref_imgs)
        B, C, H_f, W_f = feat_2d.shape
        
        # 将空间维度展平并交换维度，变为序列格式: [B, H'*W', 512]
        style_seq = feat_2d.view(B, C, -1).permute(0, 2, 1)
        
        return style_seq