#!/bin/bash
python stage2.py --output_dir ../emc_data --name N24Newsa_emc1 --dataset N24News \
--model KNet  \
--nlayers 1  \
--n_nodes 128  \
--top_k_logits 200  \
--epochs 100  \
--batch_size 128  \
--lr 1e-3 \
--gpu 0 \
--noise 0


