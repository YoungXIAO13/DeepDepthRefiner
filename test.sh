
python -W ignore testing.py \
--checkpoint model/session_ibimis_1/checkpoint_2_1.11.pth \
--val_method sharpnet \
--use_occ --no_contour \
--val_label_dir test_order_nms_pred --th 0.7

#--val_label_dir test_order_nms_pred
#--val_label_dir label --val_label_ext='-order-pix.npy' \
