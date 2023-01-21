## Implementation of the paper - <a href='https://arxiv.org/pdf/2104.08656v2.pdf'>Learning from Noisy Labels for Entity-Centric Information Extraction</a>

At first clone the repo

```bash
git clone https://github.com/giga-tech/ner-research.git
cd repo_folder
```
Then - 
```bash
cd Noisy_Label_NER
```

Then create conda environment and activate it

```bash
conda env create --name environment_name --file ./env.yml
conda activate environment_name
```

## Learning from noisy labels paper model run
Noisy labels paper default implementation uses a BERT model with a linear layer atop for classification. To run this model, run the 
train.sh 
```bash
 bash train.sh
```
<b>Results produced from this training are stored in noisy_output_bert_linear folder</b>

Banner model uses stacked bi-LSTM layers and a final CRF layer atop of the BERT encodings to generate classifications. To run this model,
modify train.sh like this - 
```bash
python main.py --is_banner
```
<b>Results produced from this training are stored in noisy_output_bert_crf folder</b>
### Banner style optimization
If you want to train banner model with the optimization style and weighted cost of banner with banner's BERT+bi-LSTM+CRF model, run the banner_train.sh
```bash
bash banner_train.sh
```
The current banner_train.sh trains the noisy label using hybrid method - banglabert + bert-multi-cased. You can specify all the models in argument --model_name_or_path by space seperated model names (huggingface model names). --is_banner argument enables banner optimization and --weighted argument enables banner weighted ce loss.
<b>Do not modify the contents of banner_train.sh - Results produced from this training are stored in noisy_output_banner_optim folder</b>
