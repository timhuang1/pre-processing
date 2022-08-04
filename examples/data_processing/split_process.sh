DATA_DIR=/apdcephfs/share_916081/timxthuang/bt_files/mono_en
PARA_DATA_DIR=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data

# export ln_count=2188068

export data_dir=/data1/paraphrase_pair_data/parallel
export file_extension=".jsonl"
# export task_name=cc_news
# export part_tail="_train_rest_add_parse"
# export prefix=$task_name"_sents"$part_tail
prefix="parallel_enzh_sents_shuf04"
corpus_file=$prefix$file_extension

# newsroom_sents_train_measure-low_div.jsonl
# msmarco_sents_measure-low_div.jsonl

export src_file=$data_dir/$corpus_file
export out_file=$data_dir/$prefix-part
export nsplit=85

ln_count=$(wc -l < $src_file)
ln_count=$(($ln_count + 1000))

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
# # Parse_only: Measure tree-diversity of sentence-pairs
# # =========
# func_name=diversity_measure
# text_cols="src_en::pred_en"
# out_filename=$prefix"_add_parse"$file_extension
# python backtrans_process.py\
#   --data_dir $data_dir\
#   --sub_files $all_splits\
#   --nsplit $nsplit\
#   --text_cols $text_cols\
#   --out_filename $out_filename\
#   --func_name $func_name\
#   --use_gpu
# # =========

# # =========
# # Compute edits: Measure tree-diversity of sentence-pairs
# # =========
# func_name=compute_tree_edit
# text_cols="ref_tree::pred_tree"
# out_filename=$prefix"_measure"$file_extension
# python backtrans_process.py\
#   --data_dir $data_dir\
#   --sub_files $all_splits\
#   --nsplit $nsplit\
#   --text_cols $text_cols\
#   --out_filename $out_filename\
#   --func_name $func_name
# # =========

# # =========
# # Selet highest-div pred result
# # =========
# func_name="pred_res_select"
# text_cols="src_para_pred"
# out_filename=$prefix"_selected"$file_extension
# python backtrans_process.py\
#   --data_dir $data_dir\
#   --sub_files $all_splits\
#   --nsplit $nsplit\
#   --text_cols $text_cols\
#   --out_filename $out_filename\
#   --func_name $func_name\

# # =========


# # =========
# # measure bow-div only
# # =========
# func_name="pred_res_select"
# text_cols="pred_en"
# out_filename=$prefix"_bow-div"$file_extension
# python backtrans_process.py\
#   --data_dir $data_dir\
#   --sub_files $all_splits\
#   --nsplit $nsplit\
#   --text_cols $text_cols\
#   --out_filename $out_filename\
#   --func_name $func_name\
#   --no_sort
# # =========


# =========
# Filter Back-translation results and re-save into multiple output files
# =========
export func_name=backtrans_filter
text_cols="src_en::pred_en"
python backtrans_process.py\
  --data_dir $data_dir\
  --sub_files $all_splits\
  --nsplit $nsplit\
  --func_name $func_name\
  --text_cols $text_cols\
  --post_action "merge_to_multiple"
  # --out_filename $task_name"_sents.jsonl"\

mv $data_dir/"post_cat_file0.jsonl" $data_dir/$prefix"_high"$file_extension
mv $data_dir/"post_cat_file1.jsonl" $data_dir/$prefix"_low"$file_extension
mv $data_dir/"post_cat_file2.jsonl" $data_dir/$prefix"_trash"$file_extension
# # =========

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