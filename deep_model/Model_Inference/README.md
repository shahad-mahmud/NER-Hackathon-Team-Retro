# Model Inference

## Directory structure

The root directory follows the below structure.

```bash
├── config  # contains config files for each type of model to run inference
├── data  # data used for inference. For testing, provide a .txt file in (see banner/test.txt for file format). Include train files for train related analysis
├── datasets  # The dataset building class for each type of model
├── evaluate  # Inference running file for each type of model
├── metrics  # Metrics calculation file for each type of model
├── output  # Output folder for each type of model. Generated predictions and corresponding analysis will be saved here.
├── utils  # Miscellanous codes needed for each type of model
├── README.md
├── main.py  # The main file that executes the inference pipeline
└── env.yaml #conda environment build file
```

## Prepare environment

```bash
conda env create --name NAME --file=env.yaml
```

### Run Inference

The lemmatizer can be used both as a python function and from CLI. To run from CLI it takes the following argument:

```text
arguments:
  config        Path to the config file
```

Run it as the following example for banner/sentence level resampling model:

```bash
python main.py --config config/noisy_hackathon.yaml
```
