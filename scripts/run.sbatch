#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclude=i8009
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --time=01:00:00
#SBATCH --account=p_scads_biollm
#SBATCH --job-name=fft_llama
#SBATCH --partition=alpha
#SBATCH --output=/data/horse/ws/chja176b-sysbio_cj/git_repos/Finetuning_LLM_Experiments/logs/taurus_logs_%x_%j.out
#SBATCH --error=/data/horse/ws/chja176b-sysbio_cj/git_repos/Finetuning_LLM_Experiments/logs/taurus_logs_%x_%j.err


ml release/23.04 GCCcore/11.3.0 Python/3.10.4 Anaconda3/2022.05 CUDA/12.0.0
source /software/rome/r23.04/Anaconda3/2022.05/etc/profile.d/conda.sh

cd ..

echo "START TIME: $(date)"
echo "Executing sbatch script $SLURM_JOB_NAME-$SLURM_JOB_ID"

conda activate /data/horse/ws/chja176b-sysbio_cj/git_repos/Finetuning_LLM_Experiments/.conda

echo "Loaded Virtual Environment with Python version:"
/data/horse/ws/chja176b-sysbio_cj/git_repos/Finetuning_LLM_Experiments/.conda/bin/python -c 'import platform; print(platform.python_version())'
echo "Pip packages:"
/data/horse/ws/chja176b-sysbio_cj/git_repos/Finetuning_LLM_Experiments/.conda/bin/pip list


export PYTHONPATH=/data/horse/ws/chja176b-sysbio_cj/git_repos/Finetuning_LLM_Experiments/.conda/bin/python:/data/horse/ws/chja176b-sysbio_cj/git_repos/Finetuning_LLM_Experiments
export WANDB=5e46d825805a997ad8e852ae5f8738d603eaa80d
export HUGGINGFACE_API_TOKEN=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
export HF_HOME=/data/horse/ws/chja176b-sysbio_cj/git_repos/Finetuning_LLM_Experiments/huggingface_home/misc
export HF_DATASETS_CACHE=/data/horse/ws/chja176b-sysbio_cj/git_repos/Finetuning_LLM_Experiments/huggingface_home/datasets

echo
export LAUNCHER="/data/horse/ws/chja176b-sysbio_cj/git_repos/Finetuning_LLM_Experiments/.conda/bin/python /data/horse/ws/chja176b-sysbio_cj/git_repos/Finetuning_LLM_Experiments/finetuning.py --config /data/horse/ws/chja176b-sysbio_cj/git_repos/Finetuning_LLM_Experiments/configs/full_finetuning/1B/8k/llama_3_2_1b.json --job_id $SLURM_JOB_ID"
echo $LAUNCHER
srun --jobid $SLURM_JOB_ID bash -c '$LAUNCHER'

echo "Done."
echo "END TIME: $(date)"