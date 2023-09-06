#!/bin/bash

set -euo pipefail

RAW_DATA=data/ted_raw/bel_eng/
BINARIZED_DATA=data/ted_binarized/bel_flores/bel_eng/
PRETRAINED_DIR=checkpoints/flores101_mm100_175M
MODEL_DIR=checkpoints/ted_bel_flores/bel_eng/
mkdir -p $MODEL_DIR

cat $PRETRAINED_DIR/language_pairs.txt \
	| tr ',' '\n' \
	| sed -r 's/^([a-z]{2,3})-[a-z]{2,3}$/\1/' \
	| sort -u > $MODEL_DIR/langs.txt

fairseq-train \
	$BINARIZED_DATA \
	--task translation_multi_simple_epoch \
	--arch transformer_wmt_en_de_big \
	--share-decoder-input-output-embed \
	--share-all-embeddings \
	--encoder-normalize-before \
	--decoder-normalize-before \
	--encoder-embed-dim 512 \
	--decoder-embed-dim 512 \
	--encoder-ffn-embed-dim 2048 \
	--decoder-ffn-embed-dim 2048 \
	--finetune-from-model $PRETRAINED_DIR/model.pt \
	--lang-dict $MODEL_DIR/langs.txt \
	--lang-pairs be-en \
	--encoder-langtok "src" \
    --decoder-langtok \
	--share-decoder-input-output-embed \
	--share-all-embeddings \
	--encoder-normalize-before \
	--max-epoch 80 \
    --patience 5 \
    --distributed-world-size 1 \
	--no-epoch-checkpoints \
	--dropout 0.3 \
	--optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	--warmup-init-lr 1e-7 --warmup-updates 1000 --lr 5e-5  \
	--criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	--max-tokens 512 \
	--update-freq 8 \
	--seed 2 \
  	--save-dir $MODEL_DIR \
	--log-interval 20 2>&1 | tee $MODEL_DIR/train.log 

# translate & eval the valid set
fairseq-generate $BINARIZED_DATA \
    --gen-subset valid \
	--task translation_multi_simple_epoch \
	--source-lang be --target-lang en \
    --path $MODEL_DIR/checkpoint_best.pt \
	--fixed-dictionary $PRETRAINED_DIR/dict.txt \
	--lang-pairs $PRETRAINED_DIR/language_pairs.txt \
	--decoder-langtok --encoder-langtok src \
    --batch-size 2 \
    --remove-bpe sentencepiece \
    --beam 5 | grep ^H | cut -c 3- | sort -n | cut -f3- > "$MODEL_DIR"/valid_b5.pred

python score.py "$MODEL_DIR"/valid_b5.pred "$RAW_DATA"/ted-dev.orig.eng \
    --src "$RAW_DATA"/ted-dev.orig.bel \
    | tee "$MODEL_DIR"/valid_b5.score