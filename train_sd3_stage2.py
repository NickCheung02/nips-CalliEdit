
import logging
import math
import os
from pathlib import Path
import traceback
import copy

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from torchvision.transforms import Resize, InterpolationMode

import transformers
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)

from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module

from models.controlnet_sd3 import SD3ControlNetModel
from models.transformer_sd3 import SD3Transformer2DModel
from models.wrapper_models import WrapperModel_SD3_ControlNet_with_Adapter
from models.adapter_models import LinearAdapterWithLayerNorm

from utils.utils import check_and_create_directory
from utils.args_utils import parse_args
from utils.sd3_utils import *
from utils.eval_utils import log_validation_with_pipeline, get_validation_dataset_and_dataloader_e2e

# ================== 修复补丁开始 ==================
import contextlib                # <--- 必须导入这个！
import accelerate.utils.other    # <--- 必须导入这个！

# 修复 PermissionError: [Errno 1] Operation not permitted
# 这一步是为了欺骗 accelerate 库，防止它去清空被锁定的环境变量
@contextlib.contextmanager
def no_op_clear_environment():
    yield

accelerate.utils.other.clear_environment = no_op_clear_environment
# ================== 修复补丁结束 ==================

logger = get_logger(__name__)

def load_transfomer(args):
    logger.info(
        f"Loading existing transformer weights from : {args.pretrained_model_name_or_path}"
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
    )
    return transformer

def load_vae(args):
    logger.info(
        f"Loading existing vae weights from : {args.pretrained_model_name_or_path}"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    return vae

# def load_controlnet(args, transformer, additional_in_channel=0, pretrained_path=None):
#     if pretrained_path:
#         logger.info(f"Loading existing controlnet weights from : {args.controlnet_model_name_or_path}")
#         controlnet = SD3ControlNetModel.from_pretrained(
#             pretrained_path, additional_in_channel=additional_in_channel
#         )
#     else:
#         logger.info("Initializing controlnet weights from transformer")
#         controlnet = SD3ControlNetModel.from_transformer(
#             transformer, num_layers=args.ctrl_layers, additional_in_channel=additional_in_channel
#         )
#     return controlnet

def load_controlnet(args, transformer, additional_in_channel=0, pretrained_path=None, interaction_mode=None):
    if pretrained_path:
        logger.info(f"Loading existing controlnet weights from : {args.controlnet_model_name_or_path}")
        # from_pretrained 通常会将额外的 kwargs 传给 __init__
        controlnet = SD3ControlNetModel.from_pretrained(
            pretrained_path, additional_in_channel=additional_in_channel, interaction_mode=interaction_mode
        )
    else:
        logger.info(f"Initializing controlnet weights from transformer (mode={interaction_mode})")
        # 手动初始化以支持 interaction_mode
        config = transformer.config
        config["num_layers"] = args.ctrl_layers
        config["additional_in_channel"] = additional_in_channel
        config["interaction_mode"] = interaction_mode # <--- 关键修改
        
        controlnet = SD3ControlNetModel(**config)
        
        # 复制权重 (手动实现 from_transformer 的逻辑)
        controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict(), strict=False)
        controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict(), strict=False)
        controlnet.context_embedder.load_state_dict(transformer.context_embedder.state_dict(), strict=False)
        controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)
        
    return controlnet

def load_text_encoders(args, class_one, class_two, class_three):
    logger.info(
        f"Loading existing text_encoder weights from : {args.pretrained_model_name_or_path}"
    )
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_3",
        revision=args.revision,
    )
    return text_encoder_one, text_encoder_two, text_encoder_three

def main(args):
    args.output_dir = os.path.join(args.output_dir, args.name)
    check_and_create_directory(args.output_dir)

    logging_dir = Path(args.output_dir, args.logging_dir)

    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if args.deepspeed:
        from configs.deepspeed_config import get_ds_plugin
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            deepspeed_plugin=get_ds_plugin(args),
            kwargs_handlers=[ddp_kwargs]
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs]
        )

    args.num_processes = accelerator.num_processes
    args.process_index = accelerator.process_index

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler
    logger.info(f"Loading scheduler from : {args.pretrained_model_name_or_path}")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

 
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

    # Load text encoder
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

    # load vae
    vae = load_vae(args)

    # load transformers
    transformer = load_transfomer(args)

    # load controlnet
    controlnet_inpaint = load_controlnet(args, transformer, additional_in_channel=1, pretrained_path=args.controlnet_model_name_or_path)

    # Initialize text controlnet, NOTE: pretrained_path = None
    # controlnet_text = load_controlnet(args, transformer, additional_in_channel=0, pretrained_path=None)
    # [找到这就行] Initialize text controlnet...
    # [修改为] 传入 interaction_mode='receiver'
    controlnet_text = load_controlnet(args, transformer, additional_in_channel=0, pretrained_path=None, interaction_mode='receiver')
    # Load ocr related
    adapter = LinearAdapterWithLayerNorm(128, 4096)

    # load pretrained text_controlnet
    wrapper_model = WrapperModel_SD3_ControlNet_with_Adapter(controlnet_text, adapter)

    # load Stage1 checkpoint
    # wrapper_model.load_state_dict(torch.load(args.controlnet_model_name_or_path2))
    # 加载 Stage 1 权重
    print(f"Loading stage 1 checkpoint from {args.controlnet_model_name_or_path2}...")
    state_dict = torch.load(args.controlnet_model_name_or_path2, map_location="cpu")

    # --------------- 修正开始 ---------------
    # 错误写法：wrapper_model.load_state_dict(state_dict, strict=False)
    # 正确写法：必须把返回值赋给 missing_keys 和 unexpected_keys 两个变量
    missing_keys, unexpected_keys = wrapper_model.load_state_dict(state_dict, strict=False) # <--- 关键修正
    # --------------- 修正结束 ---------------

    # 打印缺失的键，确认主要是 interaction_blocks
    if len(missing_keys) > 0:
        print(f"Missing keys (Expected for Stage 2 new modules): {len(missing_keys)}")
        interaction_keys = [k for k in missing_keys if "interaction_blocks" in k]
        if len(interaction_keys) > 0:
            print(f" - Contains {len(interaction_keys)} interaction_blocks keys (will be trained from scratch).")
        print(f" - Example missing keys: {missing_keys[:5]}")


    controlnet_text = wrapper_model.controlnet
    adapter = wrapper_model.adapter 
    
    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # freeze models
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # === CalliEdit-V2 训练策略 ===
    # 1. 冻结背景流 (Sender)
    controlnet_inpaint.requires_grad_(False)
    controlnet_inpaint.eval()  # 设为 eval 模式以关闭 Dropout，保证特征稳定

    # 2. 训练文字流 (Receiver)
    # 我们需要训练 adapter 和 controlnet_text (包含新的 TSIM 模块)
    adapter.requires_grad_(True)
    controlnet_text.requires_grad_(True) 
    wrapper_model.requires_grad_(True)
    wrapper_model.train()
    # ============================

    # adapter.requires_grad_(False)
    # controlnet_text.requires_grad_(False) # for frozen text_controlnet
    # wrapper_model.requires_grad_(False) 

    # controlnet_inpaint.requires_grad_(True) # for bg inpaint learning
    # controlnet_inpaint.train() # for bg inpaint learning


    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet_inpaint).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet_inpaint).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        state_dict = torch.load(args.resume_from_checkpoint, map_location='cpu')

        try:
            controlnet_inpaint.load_state_dict(state_dict)
            print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
        except Exception as e:
            print("model weights don't match the model")
            raise e

        step_info = args.resume_from_checkpoint.split('/')[-1].split('_')[0]
        if step_info.startswith('last-'):
            global_step = int(step_info[5:])
        elif step_info.isdigit():
            global_step = int(step_info)
        else:
            global_step = 0
    else:
        global_step = 0


    # Optimizer creation
    # params_to_optimize = wrapper_model.parameters()
    params_to_optimize = wrapper_model.parameters()
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    from data_utils.poster_dataset_e2e_train import Poster_Dataset
    print("Imported Poster_Dataset from poster_dataset_e2e_train.")
    train_dataset = Poster_Dataset(args=args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    import functools
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    process_caption_fn = functools.partial(
        compute_text_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        drop_rate=args.p_drop_caption,
        device=accelerator.device,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # # Prepare everything with our `accelerator`.
    # controlnet_inpaint, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     controlnet_inpaint, optimizer, train_dataloader, lr_scheduler
    # )
    # Prepare everything with our `accelerator`.
    
    # =================== 【修改 2：加入 wrapper_model】 ===================
    # 原代码：controlnet_inpaint, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(...)
    
    # 新代码：必须把 wrapper_model 也放进去，否则多卡梯度不通！
    # 注意接收返回值的顺序要和输入一致
    controlnet_inpaint, wrapper_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet_inpaint, wrapper_model, optimizer, train_dataloader, lr_scheduler
    )
    # ====================================================================
        
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=torch.float32)  # VAE need to be float32
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)
    # =================== 【修改 1：必须保持 Float32】 ===================
    # 错误：wrapper_model.to(accelerator.device, dtype=weight_dtype)
    # 正确：可训练模型必须是 FP32，Accelerate 会自动处理混合精度
    wrapper_model.to(accelerator.device, dtype=torch.float32) 
    # =================================================================
    
    # 背景 ControlNet 是冻结的，所以可以（也应该）转为 weight_dtype
    controlnet_inpaint.to(accelerator.device, dtype=weight_dtype)

    # wrapper_model.to(accelerator.device, dtype=weight_dtype)
    
    # # =================== 【新增这一行】 ===================
    # # 将背景 ControlNet 也转为半精度，解决 Input(Half) != Bias(Float) 的报错
    # controlnet_inpaint.to(accelerator.device, dtype=weight_dtype)
    # # =====================================================
    # Gradient checkpoint
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        controlnet_inpaint.enable_gradient_checkpointing()
        controlnet_text.enable_gradient_checkpointing()

    # test pipeline, dataset, dataloader, args
    if accelerator.is_main_process:
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
            controlnet_inpaint=unwrap_model(controlnet_inpaint),
            controlnet_text=controlnet_text,
            adapter=adapter,
        )
        print("load pipeline successfully!")

        test_dataloader, _ , test_args = get_validation_dataset_and_dataloader_e2e(args)
        print("load test data successfully!")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    resize = Resize(size=(args.resolution_h // 8, args.resolution_w // 8), interpolation = InterpolationMode.NEAREST, antialias=True)
           
    accelerator.wait_for_everyone()
    controlnet_inpaint.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet_inpaint):
                prompt_embeds, pooled_prompt_embeds = process_caption_fn(
                    batch["caption"]
                )

                # our custom input
                gt_im = batch['gt_im'].to(memory_format=torch.contiguous_format).to(dtype=vae.dtype, device=accelerator.device)
                controlnet_im = batch['controlnet_im'].to(memory_format=torch.contiguous_format).to(dtype=vae.dtype, device=accelerator.device)
                subject_mask = batch['subject_mask'].to(memory_format=torch.contiguous_format).to(dtype=vae.dtype, device=accelerator.device)

                # condition image latents
                with torch.no_grad():
                    # inpaint condition image
                    subject_mask = (subject_mask + 1.0) / 2.0 # [-1,1]->[0,1], 0 means need inpaint
                    cond_image_inpaint = (gt_im + 1) * subject_mask - 1
                    cond_latents_inpaint = vae.encode(cond_image_inpaint.to(dtype=vae.dtype)).latent_dist.sample()

                    cond_latents_inpaint = (cond_latents_inpaint - vae.config.shift_factor) * vae.config.scaling_factor
                    cond_latents_inpaint = cond_latents_inpaint.to(dtype=weight_dtype)
                    control_image_inpaint = torch.cat(
                        [cond_latents_inpaint, resize(subject_mask.to(dtype=weight_dtype))], dim=1
                    )  # Bx17xHxW

                    # text condition image
                    cond_latents_text = vae.encode(controlnet_im.to(dtype=vae.dtype)).latent_dist.sample()
                    cond_latents_text = (cond_latents_text - vae.config.shift_factor) * vae.config.scaling_factor
                    cond_latents_text = cond_latents_text.to(dtype=weight_dtype) # Bx16xHxW

                    # Convert images to latent space
                    latents = vae.encode(gt_im.to(dtype=vae.dtype)).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)

                # Get the text embedding for conditioning
                text_embeds = batch['text_embeds'].to(memory_format=torch.contiguous_format, dtype=weight_dtype, device=accelerator.device)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a timestep for each image
                u = compute_density_for_timestep_sampling(
                    args.weighting_scheme,
                    bsz,
                    args.logit_mean,
                    args.logit_std,
                    args.mode_scale,
                )

                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=latents.device
                )

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents
                # =================== 【修改 1：补全缺失定义】开始 ===================
                # 定义 controlnet_pooled_projections，否则后面会报错
                zero_pooled_prompt_embeds = True
                if zero_pooled_prompt_embeds:
                    controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
                else:
                    controlnet_pooled_projections = pooled_prompt_embeds
                # =================== 【修改 1：补全缺失定义】结束 ===================
                
                # controlnet(s) inpaint inference
                # [替换从这里开始] ---------------------------------------
                
                # 1. Background Stream (Sender) -> 只负责产生特征，不计算梯度
                with torch.no_grad():
                    # 调用 forward_as_sender 获取中间特征
                    control_block_samples_inpaint, bg_features_list = controlnet_inpaint.forward_as_sender(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=controlnet_pooled_projections,
                        controlnet_cond=control_image_inpaint,
                        return_dict=False,
                    )

                # 2. Text Stream (Receiver) -> 接收特征并训练
                # 调用 wrapper_model (它内部会调用 controlnet_text.forward_as_receiver)
                # 关键：传入 incoming_features=bg_features_list
                wrapper_rel = wrapper_model(
                    noisy_model_input=noisy_model_input,
                    timestep=timesteps,
                    prompt_embeds=prompt_embeds,
                    controlnet_pooled_projections=controlnet_pooled_projections,
                    controlnet_cond=cond_latents_text,
                    text_embeds=text_embeds,
                    incoming_features=bg_features_list  # <--- 注入背景特征
                )

                # [替换结束] -------------------------------------------
                # zero_pooled_prompt_embeds = True
                # if zero_pooled_prompt_embeds:
                #     controlnet_pooled_projections = torch.zeros_like(
                #         pooled_prompt_embeds
                #     )
                # else:
                #     controlnet_pooled_projections = pooled_prompt_embeds
                
                # control_block_samples_inpaint = controlnet_inpaint(
                #     hidden_states=noisy_model_input,
                #     timestep=timesteps,
                #     encoder_hidden_states=prompt_embeds,
                #     pooled_projections=controlnet_pooled_projections,
                #     controlnet_cond=control_image_inpaint,
                #     return_dict=False,
                # )[0]

                # # wrapper all learnable parameter
                # wrapper_rel = wrapper_model(
                #     noisy_model_input=noisy_model_input,
                #     timestep=timesteps,
                #     prompt_embeds=prompt_embeds,
                #     controlnet_pooled_projections=controlnet_pooled_projections,
                #     controlnet_cond=cond_latents_text,
                #     text_embeds=text_embeds,
                # )

                block_interval = (len(control_block_samples_inpaint) + 1) // len(wrapper_rel)
                control_block_samples = []
                for block_i in range(len(control_block_samples_inpaint)):
                    control_block_sample = control_block_samples_inpaint[block_i] + wrapper_rel[block_i // block_interval]
                    control_block_samples.append(control_block_sample.to(dtype=weight_dtype))

                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=control_block_samples,
                    return_dict=False,
                )[0]
            
                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                model_pred = model_pred * (-sigmas) + noisy_model_input

                weighting = compute_loss_weighting_for_sd3(
                    args.weighting_scheme, sigmas
                )

                # simplified flow matching aka 0-rectified flow matching loss
                target = latents

                # Compute regular loss.
                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # # Backpropagate
                # accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     params_to_clip = controlnet_inpaint.parameters()
                #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                # optimizer.step()
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # =================== 【修改 3：修正裁剪对象】 ===================
                    # 原代码：params_to_clip = controlnet_inpaint.parameters()
                    # 新代码：裁剪 wrapper_model 的参数
                    params_to_clip = wrapper_model.parameters()
                    # ==============================================================
                    
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                # print("backward: ", time.time() - stime)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                try:
                    accelerator.log({"loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                except Exception as e:
                    print("log error:", e)
                    traceback.print_exc()
                train_loss = 0.0
                
                # save ckpt & eval
                if accelerator.is_main_process:
                    try:
                        # save checkpoint
                        if global_step % args.checkpointing_steps == 0:
                            save_filename = '%s_net_%s.pth' % (global_step, args.name)
                            save_path = os.path.join(args.output_dir, save_filename)
                            torch.save(unwrap_model(controlnet_inpaint).state_dict(), save_path)

                        # validation
                        if  global_step % args.validation_steps == 0:
                            controlnet_inpaint.eval()
                            log_validation_with_pipeline(
                                logger=logger, pipeline=pipeline, dataloader=test_dataloader,
                                args=test_args, accelerator=accelerator, step=global_step
                            )
                            controlnet_inpaint.train()

                    except Exception as e:
                        print("validation error:", e)
                        traceback.print_exc()

                accelerator.wait_for_everyone() # wait for all processes

            # print("next step: ", time.time() - stime)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    os.environ['NCCL_MIN_NCHANNELS'] = '4'
    # 根据GPU型号，设置合适的通信参数
    cuda_device_name = torch.cuda.get_device_name()
    if 'A100' in cuda_device_name or 'A800' in cuda_device_name or 'H800' in cuda_device_name:
        os.environ['NCCL_MIN_NCHANNELS'] = '16'

    args = parse_args()
    main(args)
