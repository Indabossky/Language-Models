#!/bin/bash

module load python/3.10.4

python3 -m venv lang_envss
source lang_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
