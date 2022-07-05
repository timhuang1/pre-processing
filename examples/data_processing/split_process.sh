
export ln_count=2115834

export data_dir=/data1/paraphrase_pair_data/xsum
export file_extension=".jsonl"
export task_name=xsum
export part_tail="_train_transmart"

export corpus_file=$task_name"_sents"$part_tail$file_extension
export prefix=$task_name"_sents"$part_tail

export src_file=$data_dir/$corpus_file
export out_file=$data_dir/$prefix-part
export nsplit=75

# export ln_count=$(wc -l < $src_file)

split -da 2  -l$(($ln_count/$nsplit)) $src_file $out_file --additional-suffix=$file_extension

export all_splits=""
# for i in $(seq 0 $(($nsplit-1)));
# do
#   export all_splits=$all_splits";"$prefix-part0$i$file_extension
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


# # =========
# # Measure tree-diversity of sentence-pairs
# # =========
# func_name=diversity_measure
# text_cols="src_en::pred_en"
# out_filename=$prefix"_measure"$file_extension
# python backtrans_process.py\
#   --data_dir $data_dir\
#   --sub_files $all_splits\
#   --nsplit $nsplit\
#   --text_cols $text_cols\
#   --out_filename $out_filename\
#   --func_name $func_name\
#   --use_gpu
# # =========

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
#   --out_filename $task_name"_sents_add.jsonl"
# # =========


# # =========
# # Further filter and fix mono-en sents
# # =========
# out_filename=$prefix"_clean"$file_extension
# export func_name=fix_text_only
# python backtrans_process.py\
#   --data_dir $data_dir\
#   --sub_files $all_splits\
#   --nsplit $nsplit\
#   --func_name $func_name\
#   --out_filename $out_filename
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