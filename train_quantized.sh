CUDA_VISIBLE_DEVICES=0 python main_bwn_cos.py \
    -n bwn-preact-cos-600e_wd_1e_4 \
    -e 600 --weight-decay=1e-4
    
CUDA_VISIBLE_DEVICES=0 python main_bwn_cos.py \
    -n bwn-preact-cos-600e_wd_2e_4 \
    -e 600 --weight-decay=2e-4

# 350 cosine 50 0.001收敛
# adaptive learning rate
