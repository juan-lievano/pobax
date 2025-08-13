#!/bin/bash
#SBATCH --job-name=lightbulbs_ppo
#SBATCH --output=/nas/ucb/juanlievano/pobax/logs/lightbulbs2d_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=36GB
#SBATCH --gres=gpu:A4000:1

set -eo pipefail

echo "Running on host: $(hostname)"

export MAMBA_ROOT_PREFIX=/nas/ucb/juanlievano/miniforge3
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
export TMPDIR=/nas/ucb/juanlievano/pip_tmp

# Temporarily allow unset vars to avoid conda deactivate crash
set +u
eval "$($MAMBA_ROOT_PREFIX/bin/mamba shell hook --shell bash)"
mamba activate pobax310
set -u

cd /nas/ucb/juanlievano/pobax

echo "JAX devices:"
python -c "import jax; print(jax.devices()); print('Backend:', jax.default_backend())"


srun --nodes=1 --ntasks=1 --export=ALL,TMPDIR=$TMPDIR python -m pobax.algos.ppo \
    --env lightbulbs2d_6 \
    --platform gpu \
    --seed 2024 \
    --study_name lightbulbs2d \
    --hidden_size 64 \
    --double_critic \
    --action_concat \
    --lr 2.5e-03 \
    --ld_weight 0.25 \
    --lambda0 0.9 \
    --lambda1 0.5 \
    --n_seeds 1 \
    --num_envs 16 \
    --num_steps 128 \
    --num_minibatches 8 \
    --update_epochs 4 \
    --num_eval_envs 0 \
    --debug \
    --total_steps 10000
