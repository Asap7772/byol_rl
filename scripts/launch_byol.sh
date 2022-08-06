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
  wandb_project='byol'
  prefix='byol_project_run'
  full_run_name=$prefix'_'$index
fi

# static hyperparameters
num_epochs=1000
batch_size=256
rl_update=0
num_samples=20

# # dynamic hyperparameters
# use_both_predictions=(0)
# n_head_predictions=(0)
# num_heads=(128 256 512 1024)

# for use_both_prediction in ${use_both_predictions[@]}; do
#   for n_head_prediction in ${n_head_predictions[@]}; do
#     for num_heads in ${num_heads[@]}; do
      
#       if [ $num_heads -eq '256' ]; then
#         n_head_prediction=0
#       else 
#         n_head_prediction=1
#       fi

#       if [ $index -eq '0' ]; then
#         command="python -m byol.main_loop \
#         --experiment_mode='pretrain' \
#         --worker_mode='train' \
#         --checkpoint_root=$checkpoint_root \
#         --batch_size=$batch_size \
#         --pretrain_epochs=$num_epochs \
#         --run_name=$full_run_name \
#         --rl_update=$rl_update \
#         --num_samples=$num_samples \
#         --use_both_prediction=$use_both_prediction \
#         --n_head_prediction=$n_head_prediction \
#         --num_heads=$num_heads \
#         "
        
#         echo $command
#         if [ $dry_run = false ]; then
#           eval $command
#         fi
#       fi
#       index=$((index-1))
#     done
#   done
# done