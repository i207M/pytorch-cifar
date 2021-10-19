CUDA_VISIBLE_DEVICES=1 python main_quantized_copy.py \
    -n bwn-cos-600e-large_batch \
    -e 600 -b 384 --workers 8

CUDA_VISIBLE_DEVICES=1 python main_quantized_copy.py \
    -n bwn-cos-600e-larger_batch \
    -e 600 -b 512 --workers 8
