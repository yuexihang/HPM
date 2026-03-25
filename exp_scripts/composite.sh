data_root=${1}
lbo_path=${2}
python exp_composite.py \
    --n-hidden 128 \
    --n-heads 8 \
    --n-layers 4 \
    --lr 0.001 \
    --max_grad_norm 0.1 \
    --batch-size 16 \
    --freq_num 64 \
    --spectral_pos_embedding 0 \
    --unified_pos 0 \
    --ref 8 \
    --eval 0 \
    --epochs 5000 \
    --data_path ${data_root} \
    --lbo_path ${lbo_path} \
    --ntrain 400 \
    --save_name Composite_HPM
