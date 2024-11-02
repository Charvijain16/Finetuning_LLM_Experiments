#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclude=i8009
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --time=23:59:59
#SBATCH --account=p_scads_biollm
#SBATCH --job-name=fft_llama
#SBATCH --partition=alpha
#SBATCH --output=/data/horse/ws/chja176b-sysbio_cj/git_repos/HumanIQ_2.0/hpc_logs/taurus_logs_%x_%j.out
#SBATCH --error=/data/horse/ws/chja176b-sysbio_cj/git_repos/HumanIQ_2.0/hpc_logs/taurus_logs_%x_%j.err


ml release/23.04 GCCcore/11.3.0 Python/3.10.4 Anaconda3/2022.05 CUDA/12.0.0
source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh

cd ..

echo "START TIME: $(date)"
echo "Executing sbatch script $SLURM_JOB_NAME-$SLURM_JOB_ID"

conda activate /data/horse/ws/chja176b-sysbio_cj/environments/env_distillation

echo "Loaded Virtual Environment with Python version:"
/data/horse/ws/chja176b-sysbio_cj/environments/env_distillation/bin/python -c 'import platform; print(platform.python_version())'
echo "Pip packages:"
/data/horse/ws/chja176b-sysbio_cj/environments/env_distillation/bin/pip list


export PYTHONPATH=/data/horse/ws/chja176b-sysbio_cj/environments/env_distillation/bin/python:/data/horse/ws/chja176b-sysbio_cj/git_repos/HumanIQ_2.0:/data/horse/ws/chja176b-sysbio_cj/git_repos/HumanIQ_2.0/src:/data/horse/ws/chja176b-sysbio_cj/git_repos/HumanIQ_2.0/src/human_iq_1/app

export HUGGINGFACE_API_TOKEN=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
export HF_HOME=huggingface_home/misc
export HF_DATASETS_CACHE=huggingface_home/datasets

echo
export LAUNCHER="/data/horse/ws/chja176b-sysbio_cj/environments/env_distillation/bin/python /data/horse/ws/chja176b-sysbio_cj/git_repos/HumanIQ_2.0/src/human_iq_1/app/main.py"
echo $LAUNCHER
srun --jobid $SLURM_JOB_ID bash -c '$LAUNCHER'

echo "Done."
echo "END TIME: $(date)"