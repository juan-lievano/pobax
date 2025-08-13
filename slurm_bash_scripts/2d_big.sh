#!/bin/bash
#SBATCH --job-name=lightbulbs2d_ppo_big
#SBATCH --output=/nas/ucb/juanlievano/pobax/logs/lightbulbs2d_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=36GB
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=24:00:00
#SBATCH --nodelist=sac.ist.berkeley.edu

set -eo pipefail

echo "Running on host: $(hostname)"

export MAMBA_ROOT_PREFIX=/nas/ucb/juanlievano/miniforge3
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

# Per-job TMP so runs don't collide
export TMPDIR=/nas/ucb/juanlievano/pip_tmp/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

# JAX GPU memory behavior
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90

set +u
eval "$($MAMBA_ROOT_PREFIX/bin/mamba shell hook --shell bash)"
mamba activate pobax310
set -u

cd /nas/ucb/juanlievano/pobax

echo "JAX devices:"
python -c "import jax; print(jax.devices()); print('Backend:', jax.default_backend())"

# Bigger rollout (64x256=16384), more epochs, larger net, safer LR, higher entropy
# Unique study_name per job to avoid path collisions
srun --nodes=1 --ntasks=1 --export=ALL,TMPDIR=$TMPDIR python -m pobax.algos.ppo \
    --env lightbulbs2d_8 \
    --platform gpu \
    --seed 2024 \
    --study_name lightbulbs2d_big \
    --hidden_size 256 \
    --double_critic \
    --action_concat \
    --lr 3e-4 \
    --entropy_coeff 0.05 \
    --ld_weight 0.25 \
    --n_seeds 1 \
    --num_envs 64 \
    --num_steps 256 \
    --num_minibatches 32 \
    --update_epochs 10 \
    --num_eval_envs 0 \
    --debug \
    --total_steps 640000000
