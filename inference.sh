# python inference.py \
#  --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers/' \
#  --controlnet_model_name_or_path='./checkpoints/our_weights/scenegen_net-0415.pth' \
#  --controlnet_model_name_or_path2='./checkpoints/our_weights/textrender_net-0415.pth' \
#  --seed=42 \
#  --num_images_per_prompt=4 \
#  --use_float16


# python inference.py \
#  --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers/' \
#  --controlnet_model_name_or_path='./checkpoints/our_weights/scenegen_net-1m-0415.pth' \
#  --controlnet_model_name_or_path2='./checkpoints/our_weights/textrender_net-1m-0415.pth' \
#  --seed=42 \
#  --num_images_per_prompt=4 \
#  --use_float16

 python inference.py \
 --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers/' \
 --controlnet_model_name_or_path='./checkpoints_training/postermaker_stage1_20260317_111445/5500_net_postermaker_stage1_20260317_111445.pth' \
 --controlnet_model_name_or_path2='./checkpoints_training/postermaker_stage2_20260317_111445/11000_net_postermaker_stage2_20260317_111445.pth' \
 --font_path='./assets/fonts/FZQianLXSJW.TTF' \
 --seed 42 \
 --num_images_per_prompt 4 \
 --use_float16

#   python inference.py \
#  --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers/' \
#  --controlnet_model_name_or_path='./checkpoints/our_weights/scenegen_net-1m-0415.pth' \
#  --controlnet_model_name_or_path2='./checkpoints/our_weights/textrender_net-1m-0415.pth' \
#  --font_path='./assets/fonts/FZQianLXSJW.TTF' \
#  --seed 42 \
#  --num_images_per_prompt 4 \
#  --use_float16