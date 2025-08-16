#!/bin/bash
#SBATCH --job-name=lightbulbs_ppo
#SBATCH --output=/nas/ucb/juanlievano/pobax/logs/lightbulbs1d_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -eo pipefail

echo "Running on host: $(hostname)"

export MAMBA_ROOT_PREFIX=/nas/ucb/juanlievano/miniforge3
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
export TMPDIR=/nas/ucb/juanlievano/pip_tmp

set +u
eval "$($MAMBA_ROOT_PREFIX/bin/mamba shell hook --shell bash)"
mamba activate pobax310
set -u

cd /nas/ucb/juanlievano/pobax

echo "JAX devices:"
python -c "import jax; print(jax.devices()); print('Backend:', jax.default_backend())"

srun --nodes=1 --ntasks=1 --export=ALL,TMPDIR=$TMPDIR python -m pobax.algos.ppo \
  --env lightbulbs_40_16 \
  --hidden_size 64 \
  --double_critic \
  --action_concat \
  --total_steps 1024000000 \
  --num_envs 1024 \
  --num_steps 32 \
  --num_minibatches 8 \
  --update_epochs 3 \
  --n_seeds 1 \
  --seed 2024 \
  --lr 2.5e-03 \
  --ld_weight 0.25 \
  --lambda0 0.9 \
  --lambda1 0.5 \
  --platform gpu \
  --study_name lightbulbs
