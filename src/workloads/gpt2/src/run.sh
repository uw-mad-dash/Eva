#!/bin/bash

python -m gpt2 train --train_corpus             /datasets/wikitext-2/wiki.train.txt \
                       --eval_corpus            /datasets/wikitext-2/wiki.test.txt \
                       --vocab_path             /datasets/wikitext-2/vocab.txt \
                       --save_checkpoint_path   ckpt-gpt2.pth \
                       --save_model_path        ckpt-gpt2.pth \
                       --from_checkpoint        ckpt-gpt2.pth \
                       --batch_train            128 \
                       --batch_eval             128 \
                       --seq_len                64 \
                       --total_steps            100000 \
                       --eval_steps             10000000 \
                       --save_steps             100 \
                       --gpus                   4
