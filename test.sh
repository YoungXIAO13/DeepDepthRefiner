if [[ "$1" != "" ]]; then
    gpu="$1"
else
    gpu=0
fi

CUDA_VISIBLE_DEVICES=$gpu \
python -W ignore testing.py \
--checkpoint models/ibims_depth=1_grad=1_occ=0.01_noContour_useOcc/checkpoint_1_1.08.pth \
--val_method sharpnet \
--use_occ --no_contour \
--val_label_dir ResUnet_test_order_nms_pred_pretrain --th 0.5


#--val_label_dir test_order_nms_pred
#--val_label_dir label --val_label_ext='-order-pix.npy' \
