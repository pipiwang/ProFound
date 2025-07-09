torchrun --nproc_per_node 4 \
         --master_port=25678 \
    main_pretrain.py \
    --batch_size 4 \
    --model convnext \
    --output_dir demo/demo_output \
    --log_dir demo/demo_output \
    --mask_ratio 0.6 \
    --epochs 800 \
    --input_size "(64,224,224)" \
    --crop_spatial_size "(64,224,224)" \
    --model convnext \
    --warmup_epochs 40 \
    --blr 1e-3 \
    --norm_pix_loss \
    --weight_decay 0.05 