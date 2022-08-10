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
  wandb_project='byol_fixed'
  prefix='td_run_addsecondterm'
  full_run_name=$prefix'_'$index
fi

# static hyperparameters
num_epochs=1000
batch_size=256
rl_update=1
num_samples=20

# dynamic hyperparameters
update_types=(1 2)

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
      "

      echo $command
      if [ $dry_run = false ]; then
        eval $command
      fi
    fi
    index=$((index-1))
done
