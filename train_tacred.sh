# ./train_tacred.sh /home/compute-gdi/SpanBERT/rured/all_data_feb_07_rured.json /home/compute-gdi/SpanBERT/rured/
python /home/compute-gdi/pyutils/create_brat_tacred_dataset.py create --source=$1 --dest_dir=$2;
python ~/SpanBERT/code/run_tacred.py --do_train   --do_eval   --data_dir $2    --model "bert-base-multilingual-cased"   --from_tf False  --train_batch_size 32   --eval_batch_size 32   --learning_rate 2e-5   --num_train_epochs 10   --max_seq_length 128   --output_dir $2


