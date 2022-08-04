python -m byol.main_loop \
  --experiment_mode='pretrain' \
  --worker_mode='train' \
  --checkpoint_root='/tmp/byol_checkpoints' \
  --batch_size=256 \
  --pretrain_epochs=1000