export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=sheath_insertion \
    --checkpoint_path=first_run \
    --demo_path=/home/terry/hil-serl/src/hil-serl/examples/experiments/sheath_insertion/demo_data/sheath_insertion_20_demos_2025-12-31_11-36-39.pkl \
    --learner \