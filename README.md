# xOrder: A Model Agnostic Post-Processing Framework for Achieving Ranking Fairness While Maintaining Algorithm Utility

## Introduction
This repo includes source code and data files of experiments on benchmark datasets.  

## Requirements  
- Python 3.x  
- pytorch 1.5.0  
- scikit-learn 0.22.1  
- numpy 1.18.1  
- pandas 1.0.1  
- numba 0.48.0

## Run the experiments  
The experiment result with logistic regression classifier and xauc disparity metric on compas can be obtained with:  
`python3 run_experiments.py --dataset compas --classifier lr --eval_metric xauc`  
### Selections of datasets, ranking fairness metrics and classifiers.  
--dataset: compas, framingham  
--eval_metric: xauc, prf  
--classifier: lr(line model trained with gradient descent), rb(bipatite rankboost)  
