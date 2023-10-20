This repository contains the implementation of our paper: 
Filter-GLAT:Filter Glanced Decoder Output for Non-autoregressive Transformer
Our code is implemented based on DSLP open source code(https://github.com/chenyangh/DSLP). 

# Training:
```
python3 /home/wzc/DSLP/train.py /data2/wzc/wmt14.en-de_kd/ --source-lang en --target-lang de  --save-dir /data2/wzc/MT/WMT14/en-de/train_save_glat --eval-tokenized-bleu \
   --keep-interval-updates 3 --save-interval-updates 500 --validate-interval-updates 500 --maximize-best-checkpoint-metric \
   --eval-bleu-remove-bpe --best-checkpoint-metric bleu --log-format simple --log-interval 100 \
   --eval-bleu --eval-bleu-detok space --keep-last-epochs 5 --keep-best-checkpoints 10  --fixed-validation-seed 7 --ddp-backend=no_c10d \
   --share-all-embeddings --decoder-learned-pos --encoder-learned-pos  --optimizer adam --adam-betas "(0.9,0.98)" --lr 0.0005 \
   --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 --warmup-updates 4000 --warmup-init-lr 1e-07 --apply-bert-init --weight-decay 0.01 \
   --clip-norm 2.0 --max-update 300000 --task translation_glat --criterion glat_loss --arch glat_fliter --noise full_mask \
   --label-smoothing 0.1 --fp16  --reset-dataloader --reset-optimizer \
   --activation-fn gelu --dropout 0.1 --max-tokens 6400 --update-freq 10 --glat-mode glat \
   --length-loss-factor 0.1 --pred-length-offset --encoder-layers 6 --decoder-layers 6
```

# Evaluation:
```

```
