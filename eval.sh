python eval.py \
    --batch_size 131 \
    --input_att_dir 'data/parabu_att' \
    --input_fc_dir 'data/parabu_fc' \
    --input_json 'data/paratalk.json' \
    --input_label_h5 'data/paratalk_label.h5' \
    --language_eval 1 \
    --learning_rate 2e-5 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --max_epochs 60 \
    --rnn_type 'lstm' \
    --val_images_use 5000 \
    --save_checkpoint_every 10000 \
    --checkpoint_path 'log_xe/' \
    --id 'xe' \
    --print_freq 200


