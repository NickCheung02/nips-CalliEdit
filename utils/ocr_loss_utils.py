# utils/ocr_loss_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from models.recognizer import TextRecognizer, create_predictor
import cv2
import numpy as np
import os

class DifferentiableOCRWrapper(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        print(f"Loading OCR Reward Model from: {model_path}")
        
        # 1. 构造配置
        self.ocr_args = edict({
            'rec_image_shape': "3, 48, 320",
            'rec_batch_num': 6,
            'rec_char_dict_path': './ocr_recog/ppocr_keys_v1.txt',
            'use_fp16': False
        })
        
        # 2. 加载预测器
        self.predictor = create_predictor(model_dir=model_path, model_lang='ch')
        self.predictor.to(device)
        self.predictor.eval()
        for param in self.predictor.parameters():
            param.requires_grad = False
            
        # 3. 初始化识别器工具
        self.recognizer = TextRecognizer(self.ocr_args, self.predictor)
        
        # [新增] 内部步数计数器
        self.step_counter = 0
        
    def differentiable_crop_and_loss(self, images, batch_texts):
        """
        images: (B, C, H, W)
        batch_texts: List[Dict], where Dict contains 'content' and 'pos'
        """
        # [新增] 步数自增
        self.step_counter += 1
        # 判断是否需要保存图片 (每10步保存一次)
        is_debug_step = (self.step_counter % 500 == 0)

        device = images.device
        images_255 = (images + 1.0) * 127.5
        
        B, C, H, W = images.shape
        crop_tensors = []
        gt_strings = []
        
        num_slots = len(batch_texts) 
        
        for t_idx in range(num_slots):
            slot_data = batch_texts[t_idx]
            
            if not isinstance(slot_data, dict): continue
            contents = slot_data.get('content')
            raw_pos = slot_data.get('pos')
            
            if contents is None or raw_pos is None: continue

            # --- 智能重组坐标 (处理 default_collate 的转置问题) ---
            positions = None
            try:
                # Case A: List of 4 Tensors -> Stack
                if isinstance(raw_pos, list) and len(raw_pos) == 4 and isinstance(raw_pos[0], torch.Tensor):
                    pos_cols = [p.to(device) for p in raw_pos]
                    positions = torch.stack(pos_cols, dim=1)
                # Case B: Tensor -> Check dim
                elif isinstance(raw_pos, torch.Tensor):
                    positions = raw_pos.to(device)
                    if positions.dim() == 3 and positions.shape[1] == 1:
                        positions = positions.squeeze(1)
                else:
                    positions = torch.tensor(raw_pos, device=device)
            except Exception:
                continue
            # -------------------------------------------------
            
            for b in range(B):
                text = contents[b]
                if not text: continue 
                
                try:
                    coords = positions[b] 
                    if coords.numel() != 4: continue
                    x1, y1, x2, y2 = coords
                except Exception:
                    continue
                
                # 坐标边界保护
                x1 = torch.clamp(x1, 0, W-1)
                x2 = torch.clamp(x2, 0, W-1)
                y1 = torch.clamp(y1, 0, H-1)
                y2 = torch.clamp(y2, 0, H-1)
                
                if (x2 - x1) < 4 or (y2 - y1) < 4: continue
                
                # --- Grid Sample ---
                box_h = y2 - y1
                box_w = x2 - x1
                aspect_ratio = float(box_w / (box_h + 1e-6))
                
                target_h = 48
                target_w = int(48 * aspect_ratio)
                target_w = max(8, min(target_w, 640)) 
                
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(y1, y2, target_h, device=device),
                    torch.linspace(x1, x2, target_w, device=device),
                    indexing='ij'
                )
                
                grid_x_norm = 2.0 * grid_x / (W - 1) - 1.0
                grid_y_norm = 2.0 * grid_y / (H - 1) - 1.0
                grid = torch.stack((grid_x_norm, grid_y_norm), dim=2).unsqueeze(0)
                
                # 采样
                crop = F.grid_sample(images_255[b:b+1], grid, mode='bilinear', align_corners=True)
                
                # --- [核心修改] 保存图片逻辑 ---
                if is_debug_step:
                    try:
                        save_dir = "debug_ocr_crops"
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # 转换: Tensor (1, 3, H, W) -> Numpy (H, W, 3)
                        # 注意: 此时图片是 RGB，且值域为 [0, 255]
                        debug_img = crop.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
                        debug_img = np.clip(debug_img, 0, 255).astype(np.uint8)
                        
                        # RGB -> BGR (OpenCV 需要 BGR)
                        debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
                        
                        # 文件名: step_XXXX_b_X_字.jpg
                        # 过滤掉文件名里的非法字符
                        safe_text = "".join([c for c in text if c.isalnum() or c in (' ', '_', '-')])
                        filename = f"{save_dir}/step_{self.step_counter}_b{b}_{safe_text}.jpg"
                        
                        cv2.imwrite(filename, debug_img_bgr)
                        # print(f"[DEBUG] Saved: {filename}") # 可选：取消注释以查看日志
                        
                    except Exception as e:
                        print(f"[Warning] Save debug image failed: {e}")
                # ------------------------------

                crop_tensors.append(crop.squeeze(0))
                gt_strings.append(text)
        
        if not crop_tensors:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        processed_batch = []
        max_ratio = 320.0 / 48.0
        for img in crop_tensors:
            norm_img = self.recognizer.resize_norm_img(img, max_wh_ratio=max_ratio)
            processed_batch.append(norm_img)
            
        input_batch = torch.stack(processed_batch, dim=0)
        preds = self.predictor(input_batch)
        loss = self.recognizer.get_ctcloss(preds['ctc'], gt_strings, weight=1.0)
        
        return loss.mean()