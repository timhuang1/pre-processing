# DATA_DIR=/apdcephfs/share_916081/timxthuang/bt_files/mono_en
DATA_DIR=/Users/timhuang/Desktop/tmp_files

export task_name=msmarco_div
export data_dir=$DATA_DIR/$task_name
export in_filename=$task_name"_sents_top10k.jsonl"
export out_dir=$DATA_DIR/bpe_files
export out_filename=$task_name"_bpe.en"
export sent_key=tgt_text
export lang=english

python online_bpe.py\
  --data_dir $data_dir\
  --input $in_filename\
  --sent_key $sent_key\
  --out_dir $out_dir\
  --output $out_filename\
  --lang $lang\
