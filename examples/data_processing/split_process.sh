# DATA_DIR=/apdcephfs/share_916081/timxthuang/bt_files/mono_en
# PARA_DATA_DIR=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data
PARA_DATA_DIR=/data1/paraphrase_pair_data

export task_name=amazon_polarity
# export ln_count=2115834
export ln_count=8092133

export data_dir=$PARA_DATA_DIR/$task_name
# export data_dir=$PARA_DATA_DIR/$task_name
# export file_extension=".txt.bpe.en"
# cc_news_sents_train-add-mask.jsonl
export file_extension=".jsonl"

export part_tail="-align"
# export part_tail="_train-add-mask"
# export part_tail="_top1madd-mask"

export corpus_file=$task_name"_sents"$part_tail$file_extension
export prefix=$task_name"_sents"$part_tail

export src_file=$data_dir/$corpus_file
export out_file=$data_dir/$prefix-part
export nsplit=70

# export ln_count=$(wc -l < $src_file)

split -da 2  -l$(($ln_count/$nsplit)) $src_file $out_file --additional-suffix=$file_extension

export all_splits=""
# for i in $(seq 0 $(($nsplit-1)));
# do
#   export all_splits=$all_splits";""part0"$i".jsonl"
# done

for i in $(seq 0 9);
do
  export all_splits=$all_splits";"$prefix-part0$i$file_extension
done

for i in $(seq 10 $(($nsplit-1)));
do
  export all_splits=$all_splits";"$prefix-part$i$file_extension
done

echo $all_splits

# =========
# Filter Back-translation results and re-save into multiple output files
# =========
export func_name=backtrans_filter
python backtrans_process.py\
  --data_dir $data_dir\
  --sub_files $all_splits\
  --nsplit $nsplit\
  --func_name $func_name\
  --post_action merge_to_multiple
  # --out_filename $task_name"_sents.jsonl"\
# =========

# # =========
# # Extract well-formed sents from raw text file
# # =========
# export func_name=mono_sent_split_text_file
# python backtrans_process.py\
#   --data_dir $data_dir\
#   --sub_files $all_splits\
#   --nsplit $nsplit\
#   --func_name $func_name\
#   --out_filename $task_name"_sents.jsonl"
# # =========


# # =========
# # entity mask de-lexicalize
# # =========
# export func_name=entity_mask
# python backtrans_process.py\
#   --data_dir $data_dir\
#   --sub_files $all_splits\
#   --nsplit $nsplit\
#   --func_name $func_name\
#   --out_filename $task_name"_sents"$part_tail"-add-mask"$file_extension
# # =========

# # =========
# # seperate entity align and not-align samples 
# # =========
# export func_name=entity_mask_align
# python backtrans_process.py\
#   --data_dir $data_dir\
#   --sub_files $all_splits\
#   --nsplit $nsplit\
#   --func_name $func_name\
#   --post_action merge_to_multiple
#   # --out_filename $task_name"_sents.jsonl"\
# # =========