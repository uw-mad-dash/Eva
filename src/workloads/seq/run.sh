#!/bin/bash

TMP_DIR=./tmp
PARALLEL_TMP_DIR=./ptmp
REFERENCE=/datasets/bio/uniref50.dmnd
# REFERENCE=/datasets/bio/uniref50_1e8.dmnd
QUERY=/datasets/bio/uniprot_sprot.fasta
OUTPUT=./matches.tsv

mkdir -p $TMP_DIR
mkdir -p $PARALLEL_TMP_DIR

# ./diamond makedb --in /datasets/bio/uniref50_1e8.fasta -d uniref50_1e8.dmnd

# # # # if $PARALLEL_TMP_DIR is empty, run init
if [ -z "$(ls -A $PARALLEL_TMP_DIR)" ]; then
    ./diamond blastp -d $REFERENCE -q $QUERY --multiprocessing --mp-init --tmpdir $TMP_DIR --parallel-tmpdir $PARALLEL_TMP_DIR -b 0.21 -c 1
else
    ./diamond blastp -d $REFERENCE -q $QUERY --multiprocessing --mp-recover --tmpdir $TMP_DIR --parallel-tmpdir $PARALLEL_TMP_DIR -b 0.21 -c 1
fi

./diamond blastp -d $REFERENCE -q $QUERY -o $OUTPUT --multiprocessing --tmpdir $TMP_DIR --parallel-tmpdir $PARALLEL_TMP_DIR -b 0.21 -c 1 > diamond.log 2>&1 &
python3 monitor.py