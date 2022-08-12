index=$1
dry_run=false
debug=false

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
  wandb_project='byol_basetd'
  prefix='td_run_random_rewards'
  full_run_name=$prefix'_'$index
fi

# static hyperparameters
num_epochs=1000
batch_size=256
rl_update=1
num_samples=20

use_ensemble=0
norm_embedding=0
apply_norm=0
use_random_rewards=1

# dynamic hyperparameters
update_types=(1 2)
discounts=(0.99 0.9)
reward_scales=(1.0 0.1)

for update_type in ${update_types[@]}; do
  for discount in ${discounts[@]}; do
    for reward_scale in ${reward_scales[@]}; do
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
        --use_ensemble=$use_ensemble \
        --use_random_rewards=$use_random_rewards \
        --discount=$discount \
        --reward_scale=$reward_scale \
        "

        echo $command
        if [ $dry_run = false ]; then
          eval $command
        fi
      fi
      index=$((index-1))
    done
  done
done
