checkpoint_root='/tmp/byol_checkpoints'
rm -rf $checkpoint_root/*

python -m byol.main_loop \
  --experiment_mode='pretrain' \
  --worker_mode='train' \
  --checkpoint_root=$checkpoint_root \
  --batch_size=256 \
  --pretrain_epochs=1000 \
  --run_name='test' \
  --rl_update=1 \
  --num_samples=20
