srun --partition=boost_usr_prod \
     --account EUHPC_D27_102 \
     --gres=gpu:1 \
     --cpus-per-task=8 \
     --mem=120G \
     --time=04:00:00 \
     --pty bash