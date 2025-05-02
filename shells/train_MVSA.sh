#!/bin/bash
python train.py --batch_sz 16 --gradient_accumulation_steps 40  --savedir ./saved --name MVSA_Single --data_path ./datasets/ \
    --task MVSA_Single --task_type classification --num_image_embeds 3 --freeze_txt 5 --freeze_img 3  \
    --patience 10 --lr_patience 2 --dropout 0.1 --lr 5e-05 --max_epochs 80 --warmup 10 --seed 0 --u ratio --fusion relative2 --regressor 1

