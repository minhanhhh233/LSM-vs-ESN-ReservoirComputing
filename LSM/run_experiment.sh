#!/bin/bash

GPU=0
LOG_FILE=./experiments1.out
DATASETS=('data/ExchangeRate')
MODELS=('lsm-iaf_psc_exp-tsodyks')
PARAMETERS=experiments/parameters.json
OUTPUT=./testLsm
CSV_FILENAME=testLsm1.csv
metrics=None

# Initialize an empty array
n_train_sample=()

# Generate the sequence from 25 to 3000 with a step of 25 and store it in the array
for ((ii = 10; ii <= 6183; ii += 200)); do
    n_train_sample+=("$ii")
done
n_train_sample+=(6183)

# Set the number of times you want to run main.py
number_of_runs=1
epochs=20

for n_sample in "${n_train_sample[@]}"; do
    for ((epoch=20; epoch<=epochs; epoch+=1)); do
        for ((i=1; i<=number_of_runs; i++)); do
            # Run main.py
            
            python3 LSM/main.py --datasets ${DATASETS[@]} --models ${MODELS[@]} --n_train_sample ${n_sample} --gpu ${GPU} --parameters  $PARAMETERS --output $OUTPUT --csv_filename $CSV_FILENAME --epochss ${epoch} > $LOG_FILE 2>&1

            # Optional: Add a delay between runs (if needed)
            # sleep 1  # Adjust the time in seconds as required
        done
    done
done
