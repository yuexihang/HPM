data_root=${1}
lbo_path=${2}
python exp_bloodflow.py \
    --n-hidden 64 \
    --n-heads 8 \
    --n-layers 4 \
    --lr 0.001 \
    --max_grad_norm 0.1 \
    --batch-size 4 \
    --freq_num 64 \
    --freq_num_time 16 \
    --spectral_trans_time_length 121 \
    --spectral_pos_embedding 32 \
    --unified_pos 0 \
    --ref 8 \
    --eval 0 \
    --data_path ${data_root} \
    --lbo_path ${lbo_path} \
    --ntrain 400 \
    --save_name BloodFlow_HPM
