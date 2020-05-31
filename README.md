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

### Code
1. Neural Network [[here](main.py)]
2. Conventional ML models [[here](notebooks/test_ml.ipynb)]
3. Report [[here](docs/Heart_disease_processed_cleveland_solution_Amir.pdf)]

---
## Data
1. [processed.cleveland.data](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data) 
file, which is available from the [Data Folder](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
2. Description file 
[heart-disease.names](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names)


---
## Other Published Code 

1. https://github.com/AbdullahAlrhmoun/Heart-disease-prediction-model/blob/master/eart%2Bdisease%2Bmodelling.ipynb
2. https://www.kaggle.com/aavigan/predicting-coronary-heart-disease-non-invasively
3. https://www.kaggle.com/ronitf/predicting-heart-disease
4. https://www.kaggle.com/sharansmenon/heart-disease-pytorch-nn 
5. https://github.com/knickhill/heart-disease-classification/blob/master/part2-models.ipynb