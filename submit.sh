#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Amelia Payne, amelia28\n" > submit/team.txt
printf "Ashwini Kumar Singh, ashwini1\n" > submit/team.txt

# make predictions on example data submit it in pred.txt
python3 src/predict.py --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# make zip file
zip -r submit.zip submit
