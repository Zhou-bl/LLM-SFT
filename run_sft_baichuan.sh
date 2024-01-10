torchrun main.py \
--model_name baichuan \
--batch_size 2 \
--max_length 1024 \
--epoch 3 \
--deepspeed \
--deepspeed_config deepspeed_config/ds_config.json