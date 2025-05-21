#!/usr/bin/env bash
set -e
set -v
python src/run_app.py test predict --work_dir work --test_data $1 --test_output $2
