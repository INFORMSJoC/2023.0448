#!/bin/bash
python train.py --batch_sz 16 --gradient_accumulation_steps 40  --savedir ./saved --name CrisisMMD --data_path ./datasets/ \
    --task CrisisMMD/humanitarian --task_type classification --model emc --num_image_embeds 3 --freeze_txt 5 --freeze_img 3  \
    --patience 5 --lr_patience 2 --dropout 0.1 --lr 5e-05 --max_epochs 5 --warmup 0 --seed 0  \
    --clip_grad 0 --regressor 1