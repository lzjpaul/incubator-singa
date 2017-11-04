(1) close the shuffle function in load data (no shuffle for samples)
(2) the input is my data (label now is one column)
(3) 1~351: test, 352~1755: train

tuning
(1) port/gpuid
(2) grid search

train-paper-result-best-param
(1) need to modify lr (not changed yet, still use get_lr()), decay and momentum also

train-paper-result-load-params-best-param.py
(1) top_n changed
(2) hard code so that DRG codes are in the index_list

train-paper-result-load-params-best-param-cpu.py
(1) do not calculate true_label_prob_matrix

11-2
get_lr is replaced by lr_param

11-4
now the features are correct: 11 cases occupy the last 11 slots
