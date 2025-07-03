DATASET='risk'

CKPT=ckpt/classification.pth
OURPUTDIR=demo/demo_output
python -u demo_classfication.py \
    --batch_size 2 \
    --model 'profound_conv' \
    --ckpt_dir ${CKPT} \
    --file_name 'classfication_output' \
    --train scratch \
    --output_dir ${OURPUTDIR} \
    --log_dir ${OURPUTDIR} \
    --dataset ${DATASET} \
    --input_size "(64,224,224)" \
    --crop_spatial_size "(64,224,224)" \
    --demo True