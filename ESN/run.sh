GPU=0
LOG_FILE=./experiments${GPU}.out
DATASETS=('data/ExchangeRate')
MODELS=('esn')
PARAMETERS=experiments/parameters.json
OUTPUT=./testEsn
CSV_FILENAME=testEsn.csv

python3 main.py --datasets ${DATASETS[@]} --models ${MODELS[@]} --gpu ${GPU} --parameters  $PARAMETERS --output $OUTPUT --csv_filename $CSV_FILENAME > $LOG_FILE 2>&1 &
