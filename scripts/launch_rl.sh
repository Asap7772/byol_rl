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
  wandb_project='byol_rerun_staticrew'
  prefix='td_run_random_rewards'
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
use_random_rewards=1
n_head_prediction=1
num_heads=1024

# dynamic hyperparameters
update_types=(0 1)
discounts=(0.99 0.9)
reward_scales=(1.0)
random_reward_types=('bernoulli' 'gaussian')
static_rewards=(1 0)

for update_type in ${update_types[@]}; do
  for discount in ${discounts[@]}; do
    for reward_scale in ${reward_scales[@]}; do
      for random_reward_type in ${random_reward_types[@]}; do
        for static_reward in ${static_rewards[@]}; do
          if [ $index -eq '0' ]; then

            echo "Update Type $update_type"
            echo "Discount $discount"
            echo "Reward Scale $reward_scale"
            echo "Random Reward Type $random_reward_type"
            echo "Static Reward: $static_reward"

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
            --random_reward_type=$random_reward_type \
            --static_reward=$static_reward \
            --n_head_prediction $n_head_prediction \
            --num_heads $num_heads \
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
  done
done
