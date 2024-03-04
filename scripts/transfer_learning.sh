#!/bin/bash 

#python -m debugpy --wait-for-client --listen 0.0.0.0:54327 transfer_learning.py \
python transfer_learning.py \
--checkpoint_deterministic "checkpoints/deterministic.pth" \
--checkpoint_flipout "checkpoints/flipout.pth" \

