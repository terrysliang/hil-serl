export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export JAX_COMPILATION_CACHE_DIR="/home/terry/tmp/jax_cache" && \
python ../../train_rlpd.py "$@" \
    --exp_name=sheath_insertion \
    --checkpoint_path=first_run \
    --actor \