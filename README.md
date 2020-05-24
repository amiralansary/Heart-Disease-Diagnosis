# Heart Disease Diagnosis

## Main Task
1. Classify patients with heart failure
2. Identify correlated features

### Install
```
pip install -r requirements.txt
```

### Usage
Train

```
python main.py -t data/train.csv -v data/valid.csv --lr=0.0002 -b=20 --lr_scheduler cosine --epochs 1000 --suffix hidden_32_16
```

Evaluate

```
python main.py -e -v data/test.csv --save_results test_results.csv --lr=0.0002 -b=20 --lr_scheduler cosine --resume models/arch\[NeuralNet\]_optim\[adam\]_lr\[0.0002\]_lrsch\[cosine\]_batch\[20\]_WeightedSampling\[False\]_hidden_32_16/model_best.pth.tar
```

### Approach (options)
1. **Predictive model**
    - Use a predictive model to classify patients who have heart failure 
    - Evaluate the performance of your model and suggest which features may be useful in heart failure prediction
2. **Statistical learning** 
    - Use statistical learning (with or without dimensionality reduction) to identify the combination of feature that 
    are more likely to be associated with heart failure
    - Based on your analysis, suggest which features may be useful to predict heart failure
3. **Causal inference**
    - Use causal inference to identify the features that are more likely to have a causal connection with heart failure
    - Based on your analysis, suggest which features may be useful to predict heart failure
4. **Other approach**
    - If you would prefer not to apply the suggested approaches, then build another form of predictive/statistical 
    model that will identify relevant features. *Surprise us!*

### Solution File
Report (no more than 2 pages including plots):
1. outlining and justifying your choice of pre-processing 
2. steps you may have taken to avoid overfitting
3. parameter choices and your choice of classifier / statistical modelling compared to other approaches 
4. What can you conclude from this analysis?

If your model is time consuming to train, then please provide the model as well

### Submission
1. Your 2-page report (as doc/pdf/txt)
2. Your code, either as html/pdf (for notebooks) or share their code in a pastebin platform such as https://hastebin.com
3. Please DO NOT email us a compressed file (e.g..zip,tar.gz,etc), as these often get blocked by our antivirus

---
## First steps:
1. Download the 
[processed.cleveland.data](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data) 
file, which is available from the [Data Folder](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
2. Download the data description file 
[heart-disease.names](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names)


---
## Published solutions 

Note: I have not looked on these public solutions before I wrote mine (opposite to what I usually do before approaching 
any new problem).
This is to ensure that these published solutions will not bias my approach to solve the problem.

1. https://github.com/AbdullahAlrhmoun/Heart-disease-prediction-model/blob/master/eart%2Bdisease%2Bmodelling.ipynb
2. https://www.kaggle.com/aavigan/predicting-coronary-heart-disease-non-invasively
3. https://www.kaggle.com/ronitf/predicting-heart-disease
4. https://www.kaggle.com/sharansmenon/heart-disease-pytorch-nn 
5. https://github.com/knickhill/heart-disease-classification/blob/master/part2-models.ipynb