
python -W ignore testing.py \
--checkpoint model/session_36/checkpoint_2_1.06.pth \
--val_method sharpnet \
--use_occ \
--val_label_dir contour_pred_connectivity8

#--val_label_dir contour_pred_190801
#--val_label_dir contour_pred_connectivity8
#--val_label_dir label --val_label_ext='-order-pix.npy' \
