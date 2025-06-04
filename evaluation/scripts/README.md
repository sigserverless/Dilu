## Fine-Tuning with DeepSpeed

We use DeepSpeed based on the repository [gdGPT.git](https://github.com/CoinCheung/gdGPT.git), which is already packaged into the Docker image `lvcunchi1999/torch110cu111_deepspeed:latest`.

Follow the repository’s detailed instructions to:

1. Download the target models (e.g., **LLaMA**, **ChatGLM**).  
2. Convert the checkpoints to the format.  
3. Start fine-tuning with DeepSpeed, such as: `deepspeed /gdGPT_container/train_ds.py --config /gdGPT_container/configs/ds_config_pp_llama2.yml`