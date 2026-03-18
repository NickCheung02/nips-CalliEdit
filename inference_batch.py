import os
import json
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
import transformers
from diffusers import FlowMatchEulerDiscreteScheduler

from models.adapter_models import *
from utils.sd3_utils import *
from utils.utils import save_image, post_process
from utils.data_processor import UserInputProcessor


# inference arguments
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Batch inference of PosterMaker with JSON.")
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
    
    # === 新增批量推理所需参数 ===
    parser.add_argument("--json_path", type=str, required=True, help="Path to the test JSON file.")
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
        return None

    chars = [t['content'] for t in texts if t.get('content')]
    if not chars:
        return None

    try:
        font = ImageFont.truetype(font_path, size=target_h - 8)
    except Exception as e:
        print(f"Error loading font: {e}")
        return None

    char_crops = []
    for char in chars:
        img = Image.new('RGB', (target_h, target_h), color='white')
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (target_h - w) / 2 - bbox[0]
        y = (target_h - h) / 2 - bbox[1]
        
        draw.text((x, y), char, font=font, fill='black')
        char_crops.append(np.array(img))

    concat_img = np.concatenate(char_crops, axis=1)

    if concat_img.shape[1] > max_w:
        concat_img = concat_img[:, :max_w, :]
    else:
        pad_w = max_w - concat_img.shape[1]
        pad_img = np.ones((target_h, pad_w, 3), dtype=np.uint8) * 255
        concat_img = np.concatenate([concat_img, pad_img], axis=1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    style_tensor = transform(concat_img).unsqueeze(0) 
    return style_tensor


if __name__ == "__main__":
    args = parse_args()

    # ========================== 1. 模型加载部分 (只执行一次) ==========================
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")
    text_encoder_cls_three = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3")
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three) 
    
    tokenizer_one = transformers.CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    tokenizer_two = transformers.CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision)
    tokenizer_three = transformers.T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_3", revision=args.revision)

    vae = load_vae(args)
    transformer = load_transfomer(args)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    controlnet_inpaint = load_controlnet(args, transformer, additional_in_channel=1, num_layers=args.ctrl_layers, scratch=True)
    controlnet_text = load_controlnet(args, transformer, additional_in_channel=0, scratch=True)
    adapter = LinearAdapterWithLayerNorm(128, 4096)

    # 权重加载
    controlnet_inpaint.load_state_dict(torch.load(args.controlnet_model_name_or_path, map_location='cpu'))
    checkpoint_path = args.controlnet_model_name_or_path2
    print(f"Loading TextRenderNet from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    if 'controlnet_text' in state_dict:
        controlnet_text.load_state_dict(state_dict['controlnet_text'])
        if 'adapter' in state_dict:
            adapter.load_state_dict(state_dict['adapter'])
    else:
        controlnet_dict = {k[len('controlnet.'):]: v for k, v in state_dict.items() if k.startswith('controlnet.')}
        adapter_dict = {k[len('adapter.'):]: v for k, v in state_dict.items() if k.startswith('adapter.')}
        if controlnet_dict: controlnet_text.load_state_dict(controlnet_dict, strict=True)
        if adapter_dict: adapter.load_state_dict(adapter_dict, strict=True)

    weight_dtype = (torch.float16 if args.use_float16 else torch.float32)
    device = torch.device("cuda")

    vae.to(device=device)
    text_encoder_one.to(device=device, dtype=weight_dtype)
    text_encoder_two.to(device=device, dtype=weight_dtype)
    text_encoder_three.to(device=device, dtype=weight_dtype)
    controlnet_inpaint.to(device=device, dtype=weight_dtype)
    controlnet_text.to(device=device, dtype=weight_dtype)
    adapter.to(device=device, dtype=weight_dtype)
    
    from pipelines.pipeline_sd3 import StableDiffusion3ControlNetPipeline
    pipeline = StableDiffusion3ControlNetPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(noise_scheduler.config),
        vae=vae, transformer=transformer,
        text_encoder=text_encoder_one, tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two, tokenizer_2=tokenizer_two,
        text_encoder_3=text_encoder_three, tokenizer_3=tokenizer_three,
        controlnet_inpaint=controlnet_inpaint, controlnet_text=controlnet_text, adapter=adapter,
    )
    pipeline = pipeline.to(dtype=weight_dtype, device=device)
    data_processor = UserInputProcessor()

    # ========================== 2. 批量推理部分 ==========================
    # 加载 JSON 数据
    with open(args.json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    os.makedirs("./images/results", exist_ok=True)
    print(f"Start batch generation for {len(test_data)} items...")

    # 循环遍历每一个条目
    for idx, item in enumerate(test_data):
        # 提取当前数据的变量
        filename = item['url'].split('.')[0]  # 去掉 .jpg 后缀
        prompt = item['caption']
        texts = item['texts']

        print(f"[{idx+1}/{len(test_data)}] Generating: {filename}")

        image = np.zeros((args.resolution_h, args.resolution_w, 3), dtype=np.uint8)
        mask = np.zeros((args.resolution_h, args.resolution_w), dtype=np.uint8)

        input_data = data_processor(
            image=image,
            mask=mask,
            texts=texts,
            prompt=prompt
        )

        cond_image_inpaint = input_data['cond_image_inpaint']
        control_mask = input_data['control_mask']
        text_embeds = input_data['text_embeds']
        controlnet_im = input_data['controlnet_im']
        generator = torch.Generator(device=device).manual_seed(args.seed)

        # 实时生成 TTF 风格引导张量
        style_image_tensor = render_style_image_from_ttf(texts, args.font_path)
        if style_image_tensor is not None:
            style_image_tensor = style_image_tensor.to(device=device, dtype=weight_dtype)

        # 扩散模型推理
        results = pipeline(
            prompt=input_data['prompt'],
            negative_prompt='deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW',
            height=args.resolution_h,
            width=args.resolution_w,
            control_image=[cond_image_inpaint, controlnet_im], 
            control_mask=control_mask, 
            text_embeds=text_embeds,
            style_image=style_image_tensor,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=1.0,
            guidance_scale=args.cfg_scale,
            num_images_per_prompt=args.num_images_per_prompt,
        ).images 
        
        # 保存结果
        if len(results) == 1: 
            result_img = post_process(results[0], input_data['target_size'])
            save_image(result_img, f"./images/results/{filename}.jpg")
        else: 
            for i, result_img in enumerate(results): 
                result_img = post_process(result_img, input_data['target_size'])
                save_image(result_img, f"./images/results/{filename}_{i}.jpg")

    print("✅ All generations completed successfully!")