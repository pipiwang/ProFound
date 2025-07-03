DATASET=UCL

CKPT=ckpt/segmentation.pth
FILENAME=${model}_${seed}
OURPUTDIR=demo/demo_output
python -u demo_segmentation.py \
    --batch_size 2 \
    --model 'profound_conv' \
    --ckpt_dir ${CKPT} \
    --file_name 'segmentation_output' \
    --train scratch \
    --output_dir ${OURPUTDIR} \
    --log_dir ${OURPUTDIR} \
    --epochs 100 \
    --dataset ${DATASET} \
    --input_size "(64,224,224)" \
    --crop_spatial_size "(64,224,224)" \
    --demo True
