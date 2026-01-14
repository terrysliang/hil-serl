export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export JAX_COMPILATION_CACHE_DIR="/home/terry/tmp/jax_cache" && \
python ../../eval_record.py "$@" \
  --exp_name=sheath_insertion \
  --checkpoint_path=/home/terry/hil-serl/src/hil-serl/examples/experiments/sheath_insertion/first_run \
  --eval_checkpoint_step=63000 \
  --eval_n_trajs=10 \
