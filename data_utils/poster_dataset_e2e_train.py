import os
import json
import copy
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from models.text_embedder import FourierEmbedder
from utils.utils import *


GT_IM_SAVE_PATH = './dataset/dataset1_manu_made_1/images/'
SUBJECT_MASK_SAVE_PATH = './dataset/dataset1_manu_made_1/images/mask/'
DATA_SAMPLES_PATH = './dataset/dataset1_manu_made_1/training_data.json'

class Poster_Dataset(data.Dataset):
    def __init__(self, args, **kwargs):
        super(Poster_Dataset, self).__init__()

        with open(DATA_SAMPLES_PATH, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        self.samples = filter_samples(self.samples, check_layout)

        '''
        self.samples: List of dict
        example: 
            {
                'url': 'xxxxxx.png',
                'caption': 'xxxxxx',
                'texts': [
                    {'content': 'xxxxxx', 'pos': [xx, xx, xx, xx]},
                    {'content': 'xxxxxx', 'pos': [xx, xx, xx, xx]},
                    ...
                ],
                'logo': [
                    [xx, xx, xx, xx],
                    [xx, xx, xx, xx],
                    ...
                ],
                'texts_out': [
                    {'content': 'xxxxxx', 'pos': [xx, xx, xx, xx]},
                    {'content': 'xxxxxx', 'pos': [xx, xx, xx, xx]},
                    ...
                ],
            }
        '''


        self.backup_item = None

        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

        # image size
        self.input_size = (args.resolution_h, args.resolution_w)

        # prompt  text
        self.prompt_fix = getattr(args, "prompt", "")

        self.bg_inpaint = getattr(args, 'bg_inpaint', None)

        self.max_num_texts = getattr(args, 'max_num_texts', 7)

        self.char_padding_to_len = getattr(args, 'char_padding_to_len', None)

        self.text_feature_drop = getattr(args, "text_feature_drop", None)

        self.char_pos_encoding_dim = getattr(args, 'char_pos_encoding_dim', 32)

        self.text_pos_encoding_dim = getattr(args, 'text_pos_encoding_dim', 32)

        self.text_faeture_dim = getattr(args, 'text_faeture_dim', 64)

        self.char2feat = torch.load('./assets/char2feat_ppocr_neck64_avg.pth')
        print("dataset load ppocr64 char2feat successfully!")

        self.fourier_embedder = FourierEmbedder(num_freqs=self.text_pos_encoding_dim // (4*2)) # 4 is xywh, 2 is cos&sin

        self.len = len(self.samples)
        print(f"total trainning {self.len} samples")
        
        self.debug = args.debug

    def extract_style_image(self, poster_im, texts, target_h=48, max_w=320):
        """
        自动从原图中抠出文字，通过灰度阈值过滤背景，生成干净的白底黑字参考图
        """
        char_crops = []
        for text_info in texts:
            content = text_info.get('content', '')
            if not content: continue
            
            x1, y1, x2, y2 = [int(v) for v in text_info['pos']]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(poster_im.shape[1], x2), min(poster_im.shape[0], y2)
            
            if x2 - x1 < 5 or y2 - y1 < 5: continue
            
            # 裁剪出单个字区域
            crop = poster_im[y1:y2, x1:x2]
            
            # 二值化去背景：字是黑色，背景是山水，将较暗的像素保留为黑，其余全变白
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            # 阈值设定为 100（可根据实际图片微调，越黑的字阈值可以设越低）
            _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV) 
            
            # 创建纯白底色
            clean_crop = np.ones_like(crop) * 255
            # 将原属于黑字的区域涂黑
            clean_crop[mask == 255] = [0, 0, 0]
            
            # 等比例 Resize 到 target_h (48)
            h, w = clean_crop.shape[:2]
            if h == 0 or w == 0: continue
            new_w = max(1, int(w * (target_h / h)))
            clean_crop = cv2.resize(clean_crop, (new_w, target_h))
            char_crops.append(clean_crop)
        
        if len(char_crops) == 0:
            return np.ones((target_h, max_w, 3), dtype=np.uint8) * 255
            
        # 水平拼接所有字 (形成一整行的序列图像)
        concat_img = np.concatenate(char_crops, axis=1)
        
        # 裁剪或 Padding 到目标宽度 max_w (320)
        if concat_img.shape[1] > max_w:
            concat_img = concat_img[:, :max_w]
        else:
            pad_w = max_w - concat_img.shape[1]
            pad_img = np.ones((target_h, pad_w, 3), dtype=np.uint8) * 255
            concat_img = np.concatenate([concat_img, pad_img], axis=1)
            
        return concat_img

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.debug:
            try:
                item = self.__load_item(idx)
                return item
            except Exception as e:
                sample = self.samples[idx]
                url = sample['url']
                texts = sample['texts']
                print(idx, url, texts)
                raise e

        try:
            item = self.__load_item(idx)
            self.backup_item = item
        except (KeyboardInterrupt, SystemExit):
            raise
        except (Exception) as e:
            """Handling errors introduced by random mask generation step introduced in dataloader."""
            print('loading error: item ' + self.samples[idx]['url'])
            sample = self.samples[idx]
            url = sample['url']
            texts = sample['texts']
            print(idx, url, texts)
            if self.backup_item is not None:
                item = self.backup_item
            else:
                cur_idx = idx
                while True:
                    cur_idx = (cur_idx + 1) % self.len
                    try:
                        item = self.__load_item(cur_idx)
                        self.backup_item = item
                    except (Exception) as e: 
                        print(f'loading error: item {cur_idx}: {self.samples[cur_idx]}')
                    if item:
                        return item
                        
        return item


    def __load_item(self, idx):
        samples = self.samples[idx]
        url = samples['url'] # img relative path
        texts = copy.deepcopy(samples['texts']) 
        num_texts = len(texts)

        # Stage2: If a logo and small text can be annotated for the second stage training, it can improve the generation effect
        logos = samples['logo'] if 'logo' in samples else None
        texts_out = copy.deepcopy(samples['texts_out']) if 'texts_out' in samples else None

        optional_rel = {}

        # limit texts nums
        if num_texts > self.max_num_texts:
            # text controlnet's input needs to cache latent, so directly select top K texts
            texts, excess_texts = texts[:self.max_num_texts], texts[self.max_num_texts:]
            # filter out excess texts
            texts_out = texts_out + excess_texts if texts_out else excess_texts

            num_texts = self.max_num_texts

        # sort texts by pos(x1,y1, x2, y2)
        texts = sort_texts_by_pos(texts)

        # gt img
        # TODO: GT_IM_SAVE_PATH requires you to set it up yourself
        poster_im = read_im(url, root=GT_IM_SAVE_PATH)

        # check text pos
        poster_h, poster_w, _ = poster_im.shape
        for i in range(num_texts):
            texts[i]['pos'] = clamp_bbox_to_image(texts[i]['pos'], poster_w, poster_h)

        # === 【新增】在图片被 Resize 和 Padding 之前，抠出真实的字体风格图 ===
        style_img_np = self.extract_style_image(poster_im, texts)
        # ===================================================================

        # check logo pos
        if logos:
            for i in range(len(logos)):
                logos[i] = clamp_bbox_to_image(logos[i], poster_w, poster_h)

        # check text_out pos
        if texts_out:
            for i in range(len(texts_out)):
                texts_out[i]['pos'] = clamp_bbox_to_image(texts_out[i]['pos'], poster_w, poster_h)

        # pre-process each image' size to adapt the model's input
        poster_h, poster_w, _ = poster_im.shape
        new_h, new_w, reszie_scale = cal_resize_and_padding((poster_h, poster_w), self.input_size)
        poster_im = cv2.resize(poster_im, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        optional_rel['target_size'] = (new_h, new_w)
        optional_rel['original_size'] = (poster_h, poster_w)
       
        # pre-process each text、logo、text_out pos to adapt the model's input
        poster_h, poster_w, _ = poster_im.shape
        for i in range(num_texts):
            texts[i]['pos'] = reisize_box_by_scale(texts[i]['pos'], reszie_scale)

        if texts_out:
            for i in range(len(texts_out)):
                texts_out[i]['pos'] = reisize_box_by_scale(texts_out[i]['pos'], reszie_scale)

        if logos:
            for i in range(len(logos)):
                logos[i] = reisize_box_by_scale(logos[i], reszie_scale)


        # input img
        bg_im = poster_im.copy()

        # Stage 2: read subject mask im
        if self.bg_inpaint:
            poster_h, poster_w, _ = poster_im.shape

            # 【修改】不再读取文件，而是直接生成全黑Mask（代表没有商品，全图重绘）
            # subject_mask = read_im(url[:-4] + '.png', root=SUBJECT_MASK_SAVE_PATH)
            # subject_mask = cv2.cvtColor(subject_mask, cv2.COLOR_RGB2GRAY)
            # subject_mask = cv2.resize(subject_mask, (poster_w, poster_h), interpolation=cv2.INTER_NEAREST)
            
            # 创建一个与图片同尺寸的全0矩阵 (全黑)
            subject_mask = np.zeros((poster_h, poster_w), dtype=np.uint8)

        # Stage 1: No subject_mask required
        else: 
            subject_mask = None
        
        # mask logo in Stage 2
        if self.bg_inpaint and logos: 
            bg_im = mask_image_by_logos(bg_im, logos, 255)
            subject_mask = mask_image_by_logos(subject_mask, logos, 255) if subject_mask is not None else None

        # mask very small characters, etc in Stage 2
        if self.bg_inpaint and texts_out:
            bg_im = mask_image_by_texts(bg_im, texts_out, 255)
            subject_mask = mask_image_by_texts(subject_mask, texts_out, 255) if subject_mask is not None else None

        # gt poster
        gt_im = bg_im.copy()        
        gt_im = copy_text_to_bg(poster_im, gt_im, texts)

        # mask, text region->0, other region->1, but after transform text->-1, other->1 
        text_mask = create_mask_by_text((poster_h, poster_w), texts)

        # create empty image
        empty_image = np.zeros_like(gt_im)

        # padding image
        gt_im = pad_image_to_shape(gt_im, self.input_size, pad_value=255)
        bg_im = pad_image_to_shape(bg_im, self.input_size, pad_value=255)
        text_mask = pad_image_to_shape(text_mask, self.input_size, pad_value=255)
        empty_image = pad_image_to_shape(empty_image, self.input_size, pad_value=0)

        if self.bg_inpaint:
            optional_rel['subject_mask'] = pad_image_to_shape(subject_mask, self.input_size, pad_value=255)
    
        # Stage 1 can use fix prompt
        prompt = samples['caption'] if 'caption' in samples else self.prompt_fix

        if texts:
            with torch.no_grad():
                # Get texts feature list
                text_features, ocr_token_masks = get_char_features_by_text(texts, self.char2feat, self.char_padding_to_len)
                optional_rel['text_embeds'] = text_features
                optional_rel['text_token_masks'] = ocr_token_masks

                # ocr feature dim and pos dim
                pos_dim = self.char_pos_encoding_dim + self.text_pos_encoding_dim
                feature_dim = text_features[0].shape[-1]

                # Text feature drop
                if self.text_feature_drop:
                    text_p_drop = self.text_feature_drop ** (1/len(text_features))
                    for i in range(len(text_features)):
                        if random.random() < text_p_drop:
                            text_features[i] = torch.zeros_like(text_features[i]) # N*C
                            
                # Get char_level pos encoding
                char_positional_encoding = get_positional_encoding(self.char_padding_to_len, self.char_pos_encoding_dim) # N*32
                for i in range(len(text_features)):
                    text_features[i] = torch.cat([text_features[i], ocr_token_masks[i].unsqueeze(-1) * char_positional_encoding], dim=-1) # N*(C+32)

                # Text_level pos encoding
                for i in range(len(text_features)):
                    coords = pos2coords(texts[i]['pos']) # xyxy -> xywh
                    coords_norm = torch.tensor(normalize_coordinates(coords, self.input_size[1], self.input_size[0])) # 4
                    text_coords_embed = self.fourier_embedder(coords_norm) # 4-> 32
                    text_coords_embed = text_coords_embed.unsqueeze(0).repeat(self.char_padding_to_len, 1) # N*32
                    text_features[i] = torch.cat([text_features[i], ocr_token_masks[i].unsqueeze(-1) * text_coords_embed], dim=-1) # N*(C+32)

                # Handle ocr features(list) to a tensor
                max_token_num = self.char_padding_to_len * self.max_num_texts # to simplfiy, only no SEP
                padding_token_num = max_token_num - self.char_padding_to_len * len(text_features)
                texts_and_sep_list = []
                for i in range(len(text_features)):
                    texts_and_sep_list.append(text_features[i])
                if padding_token_num > 0:
                    texts_and_sep_list.append(torch.zeros((padding_token_num, pos_dim+feature_dim)))

                texts_all_features = torch.cat(texts_and_sep_list, dim=0) # eg. 5*16 = 80
                optional_rel['text_embeds'] = texts_all_features

                # Handle masks(list) to a tensor
                ocr_token_masks = [mask.unsqueeze(0) for mask in ocr_token_masks]
                ocr_token_masks = torch.cat(ocr_token_masks, dim=0)
                ocr_token_masks = torch.cat([ocr_token_masks, torch.zeros((self.max_num_texts - ocr_token_masks.shape[0], ocr_token_masks.shape[1]))], dim=0)
                optional_rel['text_token_masks'] = ocr_token_masks

        elif len(texts) == 0:
            # ocr feature dim and pos dim
            pos_dim = self.char_pos_encoding_dim + self.text_pos_encoding_dim
            feature_dim = self.text_faeture_dim
            max_token_num = self.char_padding_to_len * self.max_num_texts
            optional_rel['text_embeds'] = torch.zeros((max_token_num, pos_dim+feature_dim))
            optional_rel['text_token_masks'] = torch.zeros((self.max_num_texts, self.char_padding_to_len))

        # the code afer ocr feature
        if self.max_num_texts:
            texts = texts + [{'content':'', 'pos':[0,0,0,0]}]*(self.max_num_texts-num_texts)

        # rel = {
        #     'url':url,
        #     'texts': texts,
        #     'gt_im': gt_im,
        #     'bg_im': bg_im,
        #     'mask':text_mask,
        #     "caption":prompt,
        #     "num_texts":num_texts,
        #     'controlnet_im':empty_image
        # }
        rel = {
            'url':url,
            'texts': texts,
            'gt_im': gt_im,
            'bg_im': bg_im,
            'mask':text_mask,
            "caption":prompt,
            "num_texts":num_texts,
            'controlnet_im':empty_image,
            'style_image': style_img_np  # <--- 【新增】这里放 Numpy，下面的 self.transform 会自动把它转为 Tensor 并归一化 [-1, 1]
        }

        rel.update(optional_rel)


        if self.transform:
            for k,v in rel.items():
                if isinstance(v, np.ndarray):
                    rel[k] = self.transform(v)

                
        return rel 

