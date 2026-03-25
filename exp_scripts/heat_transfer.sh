data_root=${1}
lbo_input_path=${2}
lbo_output_path=${3}
python exp_heat_transfer.py \
    --n-hidden 32 \
    --n-heads 1 \
    --n-layers 4 \
    --lr 0.01 \
    --max_grad_norm 0.1 \
    --batch-size 16 \
    --freq_num 128 \
    --spectral_pos_embedding 64 \
    --domain_change_layer_idx 1 \
    --unified_pos 0 \
    --ref 8 \
    --eval 0 \
    --epochs 5000 \
    --data_path ${data_root} \
    --lbo_input_path ${lbo_input_path} \
    --lbo_output_path ${lbo_output_path} \
    --ntrain 100 \
    --save_name Heat_Transfer_HPM
