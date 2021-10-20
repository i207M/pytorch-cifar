CUDA_VISIBLE_DEVICES=0 python main_bwn.py \
    -n bwn-preact-step-large_batch \
    -e 200 -b 384 --workers 8

CUDA_VISIBLE_DEVICES=0 python main_bwn.py \
    -n bwn-preact-step \
    -e 200

CUDA_VISIBLE_DEVICES=0 python main_bwn_cos.py \
    -n bwn-preact-cos-400e \
    -e 400 --weight-decay=5e-4

CUDA_VISIBLE_DEVICES=0 python main_bc.py \
    -n bc-resnet_v1-step \
    -e 200

CUDA_VISIBLE_DEVICES=0 python main_bc.py \
    -n bc-resnet_v1-cos \
    -e 200 --weight-decay=5e-4
