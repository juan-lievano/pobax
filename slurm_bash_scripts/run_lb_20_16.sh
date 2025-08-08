#!/bin/bash
#SBATCH --job-name=lightbulbs_ppo
#SBATCH --output=logs/lightbulbs_20_16_h32_ts_4mm_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=36GB
#SBATCH --gres=gpu:A4000:1
#SBATCH --time=24:00:00
#SBATCH --nodelist=ppo.ist.berkeley.edu

set -eo pipefail

###############################################################################
# Environment bootstrap
###############################################################################

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

###############################################################################
# Run PPO experiment
###############################################################################

srun --nodes=1 --ntasks=1 --export=ALL,TMPDIR=$TMPDIR python -m pobax.algos.ppo \
    --env lightbulbs_20_16 \
    --hidden_size 32 \
    --double_critic \
    --action_concat \
    --total_steps 4000000 \
    --seed 2024 \
    --lr 2.5e-03 \
    --ld_weight 0.25 \
    --lambda0 0.9 \
    --lambda1 0.5 \
    --platform gpu \
    --study_name lightbulbs_test \
    --debug

