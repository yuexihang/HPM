data_root=${1}
python exp_irregular_darcy.py \
    --n-hidden 128 \
    --n-heads 8 \
    --n-layers 4 \
    --lr 0.001 \
    --max_grad_norm 0.1 \
    --batch-size 16 \
    --freq_num 32 \
    --spectral_pos_embedding 32 \
    --unified_pos 0 \
    --ref 8 \
    --eval 0 \
    --data_path ${data_root} \
    --ntrain 1000 \
    --save_name Irregular_Darcy_HPM
