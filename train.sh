CUDA_VISIBLE_DEVICES=1 python main.py \
    -n preact-step-large_batch \
    -e 200 -b 384 --workers 8
    
CUDA_VISIBLE_DEVICES=1 python main.py \
    -n preact-step \
    -e 200 -b 128 --workers 4

CUDA_VISIBLE_DEVICES=1 python main.py \
    -n preact-step-wd_5e_4 \
    -e 200 -b 128 --workers 4 \
    --weight-decay=5e-4

CUDA_VISIBLE_DEVICES=1 python main_cos.py \
    -n preact-cos-wd_5e_4 \
    -e 200 -b 128 --workers 4 \
    --weight-decay=5e-4
