if [[ "$1" != "" ]]; then
    gpu="$1"
else
    gpu=0
fi

CUDA_VISIBLE_DEVICES=$gpu \
python -W ignore testing.py \
--checkpoint model/session_ibimis_0/checkpoint_1_1.07.pth \
--val_method eigendnl_vgg \
--use_occ --no_contour \
--val_label_dir test_order_nms_pred --th 0.5
#--val_label_dir label --val_label_ext='-order-pix.npy'


#--val_label_dir test_order_nms_pred
#--val_label_dir label --val_label_ext='-order-pix.npy' \
