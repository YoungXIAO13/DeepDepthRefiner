if [[ "$1" != "" ]]; then
    gpu="$1"
else
    gpu=0
fi

CUDA_VISIBLE_DEVICES=$gpu \
python -W ignore testing.py \
--checkpoint ibims_depth=0_occ=0.01_change=1_noContour_useOcc_root2/checkpoint_1_1.08.pth \
--val_method sharpnet \
--use_occ --no_contour \
--val_label_dir ResUnet_test_order_nms_pred_pretrain --th 0.7


#--checkpoint experiments/ibims_depth=1_grad=1_occ=0.1_change=1_noContour_useOcc_1e-5/checkpoint_16_1.06.pth \
