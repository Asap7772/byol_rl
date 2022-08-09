index=$1
dry_run=false
debug=true

echo "Launching $index"

checkpoint_root='/home/anikaitsingh/hdd/byol_checkpoints'
mkdir -p $checkpoint_root
rm -rf $checkpoint_root/*

# wandb hyperparameters
if [ $debug = true ]; then
  echo "Debug mode"
  dry_run=false
  wandb_project='test'
  full_run_name='test'
else
  wandb_project='byol_fixed'
  prefix='td_run_normalize_embedout'
  full_run_name=$prefix'_'$index
fi

# static hyperparameters
num_epochs=1000
batch_size=256
rl_update=1
num_samples=20

use_ensemble=1
norm_embedding=0
apply_norm=0

# dynamic hyperparameters
update_types=(0)

for update_type in ${update_types[@]}; do
    if [ $index -eq '0' ]; then
      command="python -m byol.main_loop \
      --experiment_mode='pretrain' \
      --worker_mode='train' \
      --checkpoint_root=$checkpoint_root \
      --batch_size=$batch_size \
      --pretrain_epochs=$num_epochs \
      --run_name=$full_run_name \
      --rl_update=$rl_update \
      --num_samples=$num_samples \
      --update_type=$update_type \
      --wandb_project=$wandb_project \
      --use_ensemble=$use_ensemble
      "

      echo $command
      if [ $dry_run = false ]; then
        eval $command
      fi
    fi
    index=$((index-1))
done
