#!/bin/bash

#SBATCH --job-name=test_slurm_autorequeue
#SBATCH --output=test_slurm_autorequeue_ddp.log
#SBATCH --open-mode=append # don't overwrite output file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --time=00:5:00
#SBATCH --mem=48Gb
#SBATCH --cpus-per-task=8
#SBATCH --signal=SIGUSR1@60 # for pytorch-lightning auto-requeue

# setup vars
conda_env="ltp"
home_dir="/home/mila/m/mirceara"
ltp_dir="${home_dir}/lightning-transformer-pretraining"
script_path="${ltp_dir}/ltp/hf_mlm/run.py"
run_from_path=ltp_dir
bashrc_path="${home_dir}/.bashrc"


# script args
args=" \
--experiment_name=test_slurm_autorequeue_ddp \
--no_timestamp_id \
--strategy=ddp \
--num_training_steps=1000 \
--skip_eval \
--slurm_auto_requeue \
--wandb_project=ltp \
--wandb_entity='amr-amr'
"

# setup environment
cd $run_from_path
pwd; hostname; date
source ${bashrc_path}
module load anaconda/3 && activate $ltp
export PYTHONFAULTHANDLER=1

# run experiment
cmd="srun python $script_path $args"
echo $cmd
eval $cmd