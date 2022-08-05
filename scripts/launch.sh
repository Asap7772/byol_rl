index=$1

echo "Launching $index"

checkpoint_root='/home/anikaitsingh/hdd/byol_checkpoints'
mkdir -p $checkpoint_root
rm -rf $checkpoint_root/*

update_types=(0 1 2 3)
dry_run=true

for update_type in ${update_types[@]}; do
    if [ $index -eq '0' ]; then
      command="python -m byol.main_loop \
      --experiment_mode='pretrain' \
      --worker_mode='train' \
      --checkpoint_root=$checkpoint_root \
      --batch_size=256 \
      --pretrain_epochs=1000 \
      --run_name='test' \
      --rl_update=1 \
      --num_samples=20 \
      --update_type=$update_type \
      "

      echo $command
      if [ $dry_run = false ]; then
        eval $command
      fi
    fi
    index=$((index-1))
done
