DATA_DIR=/apdcephfs/share_916081/timxthuang/bt_files/mono_en

# export task_name=xsum
# export data_dir=$DATA_DIR/$task_name
# export in_filename=$task_name"_sents.jsonl"
export out_dir=$DATA_DIR/bpe_files

# declare -a task_arr=("xsum" "amazon_polarity" "cc_news" "multi_news" "trip_advisor")
declare -a task_arr=("trip_advisor" "imdb")

for task_name in "${task_arr[@]}"
do
  export data_dir=$DATA_DIR/$task_name
  export in_filename=$task_name"_sents.jsonl"
  export out_filename=$task_name"_sents.txt"
  python utils.py\
    --data_dir $data_dir\
    --jsonl_file $in_filename\
    --out_dir $out_dir\
    --out_filename $out_filename\
    --text_col text
done


# # for json to txt (for trip_advisor only)

# for task_name in "${task_arr[@]}"
# do
#   export data_dir=$DATA_DIR/$task_name
#   export in_filename=$task_name".json"
#   export out_filename=$task_name".jsonl"
#   python utils.py\
#     --data_dir $data_dir\
#     --jsonl_file $in_filename\
#     --out_dir $out_dir\
#     --out_filename $out_filename\
#     --text_col text
# done