data_root=${1}
python exp_pipe_turbulance.py \
    --n-hidden 128 \
    --n-heads 8 \
    --n-layers 8 \
    --lr 0.01 \
    --max_grad_norm 0.1 \
    --batch-size 32 \
    --freq_num 64 \
    --spectral_pos_embedding 0 \
    --unified_pos 0 \
    --ref 8 \
    --eval 0 \
    --epochs 1000 \
    --data_path ${data_root} \
    --ntrain 300 \
    --save_name Pipe_Turbulance_HPM
