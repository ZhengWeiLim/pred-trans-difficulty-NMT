## About

This project uses data from the CRITT Translation Process Research Database, and evaluates the 
extent to which surprisal and attentional features derived from a Neural Machine Translation (NMT) 
model account for reading and production times of human translators.

All surprisal and attentional feature values have been compiled and provided in `attention-norm/*`. 
See `result-analysis.ipynb` for analysis of these values that has been presented in our paper:

```
Zheng Wei Lim, Ekaterina Vylomova, Charles Kemp, and Trevor Cohn. 2023.  
Predicting Human Translation Difficulty with Neural Machine Translation. 
In Transactions of the Association for Computational Linguistics 2024 (accepted).
```


## Extracting translation surprisal and NMT attention
To reproduce values in `attention-norm/*`
1. Download CRITT-TPRDB tables following instructions [here](https://sites.google.com/site/centretranslationinnovation/tpr-db/public-studies?pli=1)
2. On bash script, define CRITT table path, output file path, source and target level as `critt_path`, `outputf`, `src_level`, `tgt_level`
3. Run 
```
python3 src/nmt_attention.py --session_file data/translog.json --source_table_dir $critt_path
--outputf $outputf --bsz 4  -translation_only --src_level $src_level --tgt_level $tgt_level 
--normalize --max
```
Note that this will download mGPT and NLLB checkpoints for estimation of monolingual and translation surprisal, and other attentional features.
###
