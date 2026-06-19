#!/bin/bash

# TODO:
# 1) Correct the transformer model so that it isolates the FiLM
# parameters in a ModuleList.
# 2) Add an argument for selecting contrastive vs MSE loss.

# List of Python scripts with their respective arguments
scripts=(
    # "train_graph.py -v FiLM -d jn -m 0 -g 0 -e 100 -l 1e-4 -b 16"
    # "train_graph.py -v FiLMLoRA -d jn -m 0 -g 0 -e 100 -l 1e-4 -b 16"
    # "train_graph.py -v LoRA -d jn -m 0 -g 0 -e 100 -l 1e-4 -b 16"
    # "train_graph.py -v HyperNetwork -d jn -m 0 -g 0 -e 100 -l 1e-4 -b 16"

    "train_bilstm.py -v HyperNetwork -d jn -m 0 -g 0 -e 100 -l 1e-4 -b 16"
    "train_token_bilstm.py -v HyperNetwork -d jn -m 0 -g 1 -e 100 -l 1e-4 -b 16"
)

# Name of the conda environment
conda_env="torch"

# Path to global conda
CONDA_SH="/opt/miniconda3/etc/profile.d/conda.sh"

for script in "${scripts[@]}"; do
    script_name=$(echo "$script" | awk '{print $1}')
    screen_name=$(basename "$script_name" .py)

    screen -dmS "$screen_name" bash -c "
        source \"$CONDA_SH\"
        conda activate \"$conda_env\"
        python $script
        exec bash
    "

    echo "Started screen '$screen_name' for script '$script'."
done