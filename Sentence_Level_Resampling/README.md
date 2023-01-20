First clone the repo

```bash
git clone https://github.com/giga-tech/ner-research.git
cd repo_folder
```

Then reate conda environment

```bash
conda env create --name environment_name --file ./Sentence_Level_Resampling/environment.yml
```

Then - 
```bash
cd Sentence_Level_Resampling
```

## NER_Adaptive_Resampling
Modify the train code in train.sh like this without changing other things (same for all methods, do not change other things)- 
```bash
 python main.py --data_dir="./Bangla-NER-Splitted-Dataset.json"  --output_dir="./sen_model_output" \
        --method sen_sample --sen_method nsCRD
```
## NER Balanced Undersampling
Modify the train code in train.sh like this without changing other things - 
```bash
python main.py --data_dir="./Bangla-NER-Splitted-Dataset.json" \
        --output_dir="./bus_model_output" \
        --method bus
```
## NER Data Augmentation
If you want to train with all the augmentation methods, modify the train code in train.sh like this - 
```bash
python main.py --data_dir="./Bangla-NER-Splitted-Dataset.json" \
        --output_dir="./dau_model_output" \
        --method dau --augmentation MR LwTR SiS SR
```

If you want to implement "Label wise token replacement" only, modify the train code in train.sh like this - 

```bash
python main.py --data_dir="./Bangla-NER-Splitted-Dataset.json" \
        --output_dir="./dau_lwtr_model_output" \
        --method dau --augmentation LwTR
```
If you want to implement "Mention Replacement" only, modify the train code in train.sh like this - 

```bash
python main.py --data_dir="./Bangla-NER-Splitted-Dataset.json" \
        --output_dir="./dau_mr_model_output" \
        --method dau --augmentation MR
```
If you want to implement "Synonym Replacement" only, modify the train code in train.sh like this - 

```bash
python main.py --data_dir="./Bangla-NER-Splitted-Dataset.json" \
        --output_dir="./dau_sr_model_output" \
        --method dau --augmentation SR
```
If you want to implement "Shuffle within segment" only, modify the train code in train.sh like this - 

```bash
python main.py --data_dir="./Bangla-NER-Splitted-Dataset.json" \
        --output_dir="./dau_sis_model_output" \
        --method dau --augmentation SiS
```

You can mix and match any combination of augmentations