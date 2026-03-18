python inference_batch.py \
    --pretrained_model_name_or_path /path/to/sd3 \
    --controlnet_model_name_or_path /path/to/inpaint_ckpt \
    --controlnet_model_name_or_path2 /path/to/your_trained_textrender_ckpt \
    --json_path ./path/to/your_test_data.json \
    --font_path ./assets/fonts/YourCalligraphyFont.ttf \
    --use_float16