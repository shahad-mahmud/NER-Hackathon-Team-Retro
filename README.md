# Named Entity Recognition

## Installation

To use this repository install required packages using the following command:

```bash
pip install -r requirements.txt
```

## Feature based model

We used CRF as the feature based model. To train and get prediction on the test set, put the data files in the `data/raw` directory. Then put the paths in the `configs/ml_model.yaml` file. Once done, the training can be started with the following command:

```bash
python feature_train.py configs/ml_model.yaml
```

It will train and put the prediction result on test data in the `data/results/feature_model.txt` file.