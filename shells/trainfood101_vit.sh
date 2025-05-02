#!/bin/bash
python train.py --batch_sz 16 --gradient_accumulation_steps 40  --savedir ./saved --name food101 --data_path ./datasets/ \
    --task food101 --task_type classification --num_image_embeds 3 --freeze_txt 5 --freeze_img 3  \
    --patience 5 --lr_patience 2 --dropout 0.1 --lr 5e-05 --max_epochs 80 --warmup 0 --seed 0 \
    --transform vit --lr_detail y --vision_model vit --LOAD_SIZE 384 --FINE_SIZE 384 \
    --clip_grad 0.1 --regressor 1
