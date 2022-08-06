checkpoint_root='/home/anikaitsingh/hdd/byol_checkpoints'
mkdir -p $checkpoint_root
rm -rf $checkpoint_root/*


num_epochs=1000
batch_size=256
rl_update=0
full_run_name='rerun_byol_base'

command="python -m byol.main_loop \
        --experiment_mode='pretrain' \
        --worker_mode='train' \
        --checkpoint_root=$checkpoint_root \
        --batch_size=$batch_size \
        --pretrain_epochs=$num_epochs \
        --run_name=$full_run_name \
        --rl_update=$rl_update"

echo $command
eval $command