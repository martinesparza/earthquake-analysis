#!/bin/bash

# Activate your virtual environment if needed
source ~/path/to/your/venv/bin/activate

# Run scripts sequentially
python3 M062_2025_03_19_14_00_rnn.py
python3 M062_2025_03_20_14_00_rnn.py
python3 M062_2025_03_21_14_00_rnn.py

echo "✅ All RNN scripts completed!"
