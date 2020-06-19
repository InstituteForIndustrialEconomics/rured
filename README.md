Annotated files are in `annotation_files.zip` archive

the annotation.conf is used for brat (https://github.com/nlplab/brat)


So you may need to copy `annotation_files.zip` contents into `your_brat_folder/data`


annotation.conf should also be copied into `your_brat_folder/data`


`json_data.zip` contains posprocessed dataset with train/test/dev split


see `train_tacred.sh` for usage example


`convert_brat_to_tacred.py` is used to transform BRAT-files to the TACRED-format


`./create_brat_tacred_dataset.py` is used for train/dev/test split and for generating negative relation examples


we use `SpanBERT` for model training (`./SpanBERT/code/run_tacred.py`)

