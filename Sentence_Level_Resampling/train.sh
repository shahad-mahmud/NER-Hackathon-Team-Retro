cd lexical_db_bangla
python setup.py install
cd ..
python main.py --data_dir "../data/processed" \
        --output_dir "./banner_train_sen_sample" --num_train_epochs 30 