# session_3_19 UNet

python testing.py \
--checkpoint model/session_18/checkpoint_last.pth \
--val_method sharpnet \
--cat_all \
--val_label_dir label --val_label_ext='-order-pix.npy' \
