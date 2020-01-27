if [[ "$1" != "" ]]; then
    gpu="$1"
else
    gpu=0
fi

CUDA_VISIBLE_DEVICES=$gpu \
python -W ignore testing.py \
--checkpoint experiments/ibims_depth=1_grad=1_occ=0.1_change=1_noContour_useOcc_1e-5/checkpoint_16_1.06.pth \
--val_method eigen \
--use_occ --no_contour \
--val_label_dir ResUnet_test_order_nms_pred_pretrain --th 0.7


#--val_label_dir test_order_nms_pred
#--val_label_dir label --val_label_ext='-order-pix.npy' \
