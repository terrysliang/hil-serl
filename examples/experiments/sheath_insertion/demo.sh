export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../eval_demo.py "$@" \
  --exp_name=sheath_insertion \
  --checkpoint_path=/home/terry/hil-serl/src/hil-serl/examples/experiments/sheath_insertion/first_run \
  --eval_checkpoint_step=30000 \
  --eval_n_trajs=1 \
  --do_pickup=True \
