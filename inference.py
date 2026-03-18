import os
import argparse

import cv2
import numpy as np
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms  # <--- 新增
import transformers
from diffusers import FlowMatchEulerDiscreteScheduler

from models.adapter_models import *
from utils.sd3_utils import *
from utils.utils import save_image, post_process
from utils.data_processor import UserInputProcessor


# inference arguments
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of PosterMaker inference.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--controlnet_model_name_or_path2", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--resolution_h", type=int, default=1024)
    parser.add_argument("--resolution_w", type=int, default=1024)

    # number of SD3 ControlNet Layers
    parser.add_argument("--ctrl_layers", type=int, default=23,help="control layers",)
    
    # inference
    parser.add_argument('--num_inference_steps', type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="classifier-free guidance scale")
    parser.add_argument("--erode_mask", action='store_true')
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--use_float16", action='store_true')
    parser.add_argument("--font_path", type=str, default=None, help="Path to the .ttf font file for style guidance.")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def render_style_image_from_ttf(texts, font_path, target_h=48, max_w=320):
    """
    根据给定的 texts 和 ttf 字体，动态渲染出一张 [48, 320] 的白底黑字风格图
    """
    if font_path is None or not os.path.exists(font_path):
        print(f"Warning: Font path {font_path} not found or not provided. Generating without style injection.")
        return None

    # 提取所有要生成的字符
    chars = [t['content'] for t in texts if t.get('content')]
    if not chars:
        return None

    # 加载字体 (留一点边距，所以字号略小于 target_h)
    try:
        font = ImageFont.truetype(font_path, size=target_h - 8)
    except Exception as e:
        print(f"Error loading font: {e}")
        return None

    char_crops = []
    for char in chars:
        # 创建纯白底色的单字方块
        img = Image.new('RGB', (target_h, target_h), color='white')
        draw = ImageDraw.Draw(img)
        
        # 计算文字居中的位置
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (target_h - w) / 2 - bbox[0]
        y = (target_h - h) / 2 - bbox[1]
        
        # 写入黑字
        draw.text((x, y), char, font=font, fill='black')
        char_crops.append(np.array(img))

    # 水平拼接所有字
    concat_img = np.concatenate(char_crops, axis=1)

    # 裁剪或 Padding 到目标宽度 max_w (320)
    if concat_img.shape[1] > max_w:
        concat_img = concat_img[:, :max_w, :]
    else:
        pad_w = max_w - concat_img.shape[1]
        pad_img = np.ones((target_h, pad_w, 3), dtype=np.uint8) * 255
        concat_img = np.concatenate([concat_img, pad_img], axis=1)

    # 转为 Tensor 并归一化到 [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    style_tensor = transform(concat_img).unsqueeze(0) # 增加 Batch 维度: [1, 3, H, W]
    return style_tensor

if __name__ == "__main__":
    # parse arguments
    args = parse_args()

    # load text encoders
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        args, text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    ) 
    # Load tokenizers
    tokenizer_one = transformers.CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = transformers.CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = transformers.T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # load vae
    vae = load_vae(args)
    # load sd3
    transformer = load_transfomer(args)
    # load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    # create SceneGenNet
    controlnet_inpaint = load_controlnet(args, transformer, additional_in_channel=1, num_layers=args.ctrl_layers, scratch=True)
    # create TextRenderNet
    controlnet_text = load_controlnet(args, transformer, additional_in_channel=0, scratch=True)
    # load adapter
    adapter = LinearAdapterWithLayerNorm(128, 4096)

    controlnet_inpaint.load_state_dict(torch.load(args.controlnet_model_name_or_path, map_location='cpu'))
    # textrender_net_state_dict = torch.load(args.controlnet_model_name_or_path2, map_location='cpu')
    # controlnet_text.load_state_dict(textrender_net_state_dict['controlnet_text'])
    # adapter.load_state_dict(textrender_net_state_dict['adapter'])
    # 加载权重文件
    checkpoint_path = args.controlnet_model_name_or_path2
    print(f"Loading TextRenderNet from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # ================= 兼容加载逻辑开始 =================
    if 'controlnet_text' in state_dict:
        # 【情况A：官方预训练模型】
        # 结构为嵌套字典: {'controlnet_text': state_dict, 'adapter': state_dict}
        print(">>> 检测到官方模型格式 (Nested Dict)，正在加载...")
        controlnet_text.load_state_dict(state_dict['controlnet_text'])
        if 'adapter' in state_dict:
            adapter.load_state_dict(state_dict['adapter'])
            
    else:
        # 【情况B：用户自训练模型】
        # 结构为扁平字典 (WrapperModel): {'controlnet.xxx': ..., 'adapter.xxx': ...}
        print(">>> 检测到自训练模型格式 (Flat Wrapper)，正在自动拆分加载...")
        controlnet_dict = {}
        adapter_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('controlnet.'):
                # 去掉前缀 'controlnet.'
                new_key = key[len('controlnet.'):] 
                controlnet_dict[new_key] = value
            elif key.startswith('adapter.'):
                # 去掉前缀 'adapter.'
                new_key = key[len('adapter.'):]
                adapter_dict[new_key] = value
        
        # 加载分离后的权重
        if len(controlnet_dict) > 0:
            controlnet_text.load_state_dict(controlnet_dict, strict=True)
            print(f" - ControlNet: {len(controlnet_dict)} keys loaded.")
        else:
            print("Warning: No controlnet weights found in checkpoint!")

        if len(adapter_dict) > 0:
            adapter.load_state_dict(adapter_dict, strict=True)
            print(f" - Adapter: {len(adapter_dict)} keys loaded.")
        else:
            print("Warning: No adapter weights found in checkpoint!")
    # ================= 兼容加载逻辑结束 =================

    # set device and dtype
    weight_dtype =  (torch.float16 if args.use_float16 else torch.float32)
    device = torch.device("cuda")

    # move all models to device
    vae.to(device=device)
    text_encoder_one.to(device=device, dtype=weight_dtype)
    text_encoder_two.to(device=device, dtype=weight_dtype)
    text_encoder_three.to(device=device, dtype=weight_dtype)
    controlnet_inpaint.to(device=device, dtype=weight_dtype)
    controlnet_text.to(device=device, dtype=weight_dtype)
    adapter.to(device=device, dtype=weight_dtype)
    
    # load pipeline
    from pipelines.pipeline_sd3 import StableDiffusion3ControlNetPipeline
    pipeline = StableDiffusion3ControlNetPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(
            noise_scheduler.config
            ),
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        text_encoder_3=text_encoder_three,
        tokenizer_3=tokenizer_three,
        controlnet_inpaint=controlnet_inpaint,
        controlnet_text=controlnet_text,
        adapter=adapter,
    )

    pipeline = pipeline.to(dtype=weight_dtype, device=device)

    # user input processor
    data_processor = UserInputProcessor()

    # ========================== 修改开始 ==========================
    # 1. 定义保存文件名的前缀 (因为没有输入文件了，自己起个名)
    filename = 'demo_shanshui_001'

    # 2. 动态生成画布 (完全尊重 args 的分辨率参数)
    # 生成一张全黑的底图 [H, W, 3]
    image = np.zeros((args.resolution_h, args.resolution_w, 3), dtype=np.uint8)

    # 3. 动态生成 Mask
    # 全黑 Mask (数值为0) 代表全图重绘，即“生成背景”
    mask = np.zeros((args.resolution_h, args.resolution_w), dtype=np.uint8)
    
    # 4. 提示词 (Prompt) - 保持你写的
    prompt = "This exquisite Chinese landscape painting captures a serene rural scene with delicate brushstrokes and harmonious composition. The theme revolves around a tranquil village nestled amidst lush greenery, with willow trees framing the foreground and distant mountains enveloped in mist. The main elements include blooming peach trees, quaint houses, and figures engaged in daily activities, all meticulously arranged to create depth. The color palette is soft and natural, dominated by earthy tones and gentle hues of pink and green, evoking a sense of peace and harmony with nature."
    
    # 5. 文字布局 (Texts) - 保持你写的竖排布局
    texts = [
                {"content": "春", "pos": [30, 205, 142, 317]},
                {"content": "色", "pos": [30, 331, 142, 443]},
                {"content": "向", "pos": [30, 457, 142, 569]},
                {"content": "明", "pos": [30, 583, 142, 695]},
                {"content": "归", "pos": [30, 709, 142, 821]},
            ]

    # 注意：这里不需要再调用 cv2.imread 了，因为 image 和 mask 已经在上面生成了
    # ========================== 修改结束 ==========================

    # preprocess single user input
    input_data = data_processor(
        image=image,
        mask=mask,
        texts=texts,
        prompt=prompt
    )

    # pipeline input
    cond_image_inpaint = input_data['cond_image_inpaint']
    control_mask = input_data['control_mask']
    prompt = input_data['prompt']
    text_embeds = input_data['text_embeds']
    controlnet_im = input_data['controlnet_im']
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # === 【新增】生成风格引导张量 ===
    style_image_tensor = render_style_image_from_ttf(texts, args.font_path)
    if style_image_tensor is not None:
        style_image_tensor = style_image_tensor.to(device=device, dtype=weight_dtype)
    # ================================

    # inference
    results = pipeline(
        prompt=prompt,
        negative_prompt='deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW',
        height=args.resolution_h,
        width=args.resolution_w,
        control_image=[cond_image_inpaint, controlnet_im],  # B, C, H, W
        control_mask=control_mask,  # B,1,H,W
        text_embeds=text_embeds, # B, L, C
        style_image=style_image_tensor,  # <--- 【新增】把风格图传给 pipeline
        num_inference_steps=28, # number of diffusion steps
        generator=generator,
        controlnet_conditioning_scale=1.0,
        guidance_scale=5.0, # classifier-free guidance scale
        num_images_per_prompt=args.num_images_per_prompt, # number of images to generate for each user input
    ).images # return a list of PIL.Image
    

    # save result
    os.makedirs("./images/results", exist_ok=True) # <--- 加上这一行自动建文件夹
    # save result
    if len(results) == 1: 
        image = results[0] # num_images_per_prompt == 1
        image = post_process(image, input_data['target_size'])
        output_path = f"./images/results/{filename}.jpg"
        save_image(image, output_path)
        print(f"Saved image: {output_path}")
    else: 
        for i, image in enumerate(results): # num_images_per_prompt > 1
            image = post_process(image, input_data['target_size'])
            output_path = f"./images/results/{filename}_{i}.jpg"
            save_image(image, output_path)
            print(f"Saved image: {output_path}")
