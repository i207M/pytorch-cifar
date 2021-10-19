CUDA_VISIBLE_DEVICES=1 python main_bc.py \
    -n bc-cos-preact-large_batch \
    -e 200 -b 384 --workers 8

CUDA_VISIBLE_DEVICES=1 python main_bc.py \
    -n bc-cos-preact-small_batch \
    -e 200 -b 128 --workers 4
