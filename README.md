# SentencePersonality

### This package computes personality traits, as described in Big5 model, from myPersonality dataset.

#### The pipeline is the following:
- sentence encoding 
- personality traits regression

#### The logical steps to reproduce the experiment

1. encode_and_map_sentences.py
    - input: 
        * pretrained_bert/multi_cased_L-12_H-768_A-12
        * spacymoji vocabulary
        * myPersonalitySmall/statuses_unicode.txt 
        * myPersonalitySmall/big5labels.txt
    - output:
        * train_whole_lines.csv 
        * lines_skipped.csv
2. mse.py
    - input:
        * cls_table.csv
    - output:
        * predictions.txt
        * mse.txt
3. distributions.py
    - input:
        * predictions.py
        * predictions_eng.py  //this file is used for comparison, change bert model model in previous steps
        * lines_skipped.py
        * train_whole_lines.csv
    - output:
        * data distributions as images
        * kullback leibler divergences between multilinglual model and english model

#### If you want to obtain Carducci et al. results take a look here:
    * https://github.com/D2KLab/twitpersonality

#### If you want to reproduce scores by IBM Personality Insights:
    * ibm_insights_script.py
    * format the output to compare data distributions
    
