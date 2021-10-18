CUDA_VISIBLE_DEVICES=0 python main.py \
    -e 200 -b 384 --workers 8 \
    --weight-decay 5e-4
