# ./train_tacred.sh ./all_data.json ./rured
python convert_brat_to_tacred.py "annotation_files/*.ann"
echo "Converted to Tacred-format"
python ./create_brat_tacred_dataset.py create --source='./all_data.json' \
                                              --dest_dir='./rured';
echo "Generated the Dataset"
python ./SpanBERT/code/run_tacred.py --do_train \
                                     --do_eval \
                                     --data_dir "./rured" \
                                     --model "bert-base-multilingual-cased" \
                                     --from_tf False \
                                     --train_batch_size 32 \
                                     --eval_batch_size 32 \
                                     --learning_rate 2e-5 \
                                     --num_train_epochs 10 \
                                     --max_seq_length 128 \
                                     --output_dir $2


