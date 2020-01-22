#H1 SentencePersonality

#H3 This package computes personality traits, as described in Big5 model, from myPersonality dataset.

#H4 The pipeline is the following:
- sentence encoding 
- personality traits regression

#H4 The logical steps to reproduce the experiment

1. encode_and_map_sentences.py
    ..* input: 
        ..1. pretrained_bert/multi_cased_L-12_H-768_A-12
        ..2. spacymoji vocabulary
        ..3. myPersonalitySmall/statuses_unicode.txt 
        ..4. myPersonalitySmall/big5labels.txt
    ..* output:
        ..1. train_whole_lines.csv 
        ..2. lines_skipped.csv
2. mse.py
    ..* input:
        ..1.cls_table.csv
    ..* output:
        a)predictions.txt
        b)mse.txt
3. distributions.py
    ..* input:
        ..1. predictions.py
        ..2. predictions_eng.py  //this file is used for comparison, change bert model model in previous steps
        ..3. lines_skipped.py
        ..4. train_whole_lines.csv
    output:
        ..5. data distributions as images
        ..6. kullback leibler divergences between multilinglual model and english model

#H4 If you want to obtain Carducci et al. results take a look here:
    * https://github.com/D2KLab/twitpersonality

#H4 If you want to reproduce scores by IBM Personality Insights:
    * ibm_insights_script.py
    * format the output to compare data distributions
    