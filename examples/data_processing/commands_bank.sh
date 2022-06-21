# ==========
# Commands recipe:
#   - Merging multiple txts into a jsonl file (e.g., input and pred output for transart)
#   - Transform jsonl file to pure-text txt based one a specific key(s) 
#   - Fix transmart outputs symbol fluency (necessary for back-translation data building)
#   - Merge jsonl and txt into sentence pair (.jsonl format)
#   - Re-name jsonl file fields
#   - Hf model/dataset download and re-save
#   - Transit files to/from BPE server
#   - Extract jsonl fields into txt
#   - Extract well-formed sents from raw text file
#   - Transform jsonl file to txt sentence pair for embedding-based word align
#   - Merging entity mask and word align
# ==========

# # ==========
# # Extract jsonl fields into txt
# # ==========
# export jsonl_data_dir=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data/cc_news
# # export txt_data_dir=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data/cc_news
# out_dir=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data/xsum/subset/
# jsonl_file=xsum_sents_anno_100k_trans.jsonl
# out_filename=xsum_sents_anno_100k_wTrans.txt
# python utils.py \
#   --data_dir $jsonl_data_dir \
#   --jsonl_data_dir $jsonl_data_dir\
#   --jsonl_file $jsonl_file \
#   --out_dir $jsonl_data_dir\
#   --out_filename $out_filename\
#   --func_name jsonl_to_txt
# # ==========

# # ==========
# # Fix transmart outputs symbol fluency (necessary for back-translation data building)
# # ==========
# export SCRIPT_DIR=/apdcephfs/share_916081/timxthuang/scripts
# export task=trip_advisor
# for i in $(seq 0 2);
# do
#   # /apdcephfs/share_916081/timxthuang/paraphrase_pair_data/amazon_polarity/en_bpe/amazon_polarity_sents-part00.txt.bpe.en
#   # i=2
#   # export type_dir=transmart_zh2en
#   type_dir=en_bpe
#   # export part_tail=-part0$i"-en2zh-zh2en"
#   export part_tail=-part0$i
#   echo $part_tail
#   export data_dir=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data/$task/$type_dir
#   export in_filename=$task"_sents$part_tail.txt.bpe.en"
#   export fix_filename=$task"_sents$part_tail-fix.txt.bpe.en"
#   # first do BPE decoding
#   python utils.py \
#     --data_dir $data_dir \
#     --txt_files $in_filename\
#     --out_dir $data_dir\
#     --out_filename $fix_filename\
#     --text_col src_en \
#     --func_name BPE_decode

#   perl $SCRIPT_DIR/detokenizer.perl < $data_dir/$fix_filename > $data_dir/$fix_filename".tmp"
#   # rm $data_dir/$in_filename
#   mv $data_dir/$fix_filename".tmp" $data_dir/$fix_filename
# done
# # ==========

# # ==========
# # Transit files to/from BPE server
# # ==========

# # for i in $(seq 0 3);
# # do
# #   part_tail=-part0$i
# #   DATA_DIR=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data/$task/en2zh_m2m100_418M
# #   filename=$task"_sents"$part_tail"-en2zh.txt"
# #   bpe_server_dir=/data1/tokenize/bpe_files
# #   scp $DATA_DIR/$filename root@9.146.172.40:$bpe_server_dir/
# # done

# task=xsum
# PARA_DIR=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data
# data_dir=$PARA_DIR/$task

# # type_dir=en_bpe
# # data_dir=$PARA_DIR/$task/$type_dir

# bpe_server_dir=/data1/tokenize/bpe_files/timxthuang_workspace
# # for i in $(seq 0 3);
# # do
# # done
# # xsum_sents-part03-en2zh.txt
# # part_tail=-part03-en2zh
# part_tail="_train-pred_en"
# filename=$task"_sents"$part_tail".txt.token.bpe"

# # scp $data_dir/$filename root@9.146.172.40:$bpe_server_dir/
# # xsum_sents_train-pred_en.txt
# # filename=$task"_sents"$part_tail".txt.token.bpe"
# # scp $data_dir/$filename root@9.146.172.40:$bpe_server_dir/$task
# scp root@9.146.172.40:$bpe_server_dir/$task/$filename $data_dir
# # scp root@9.146.172.40:$bpe_server_dir/$task/$filename $data_dir
# mv $data_dir/$filename $data_dir/$task"_sents"$part_tail".bpe.en"
# # mv $transmart_data_path/$task_name/$prefix-part0$i"-en2zh.txt.token.bpe" $transmart_data_path/$task_name/$prefix-part0$i"-en2zh.txt.bpe.zh"


# part_tail="_train-src_en"
# filename=$task"_sents"$part_tail".txt.token.bpe"

# # scp $data_dir/$filename root@9.146.172.40:$bpe_server_dir/
# # xsum_sents_train-pred_en.txt
# # filename=$task"_sents"$part_tail".txt.token.bpe"
# # scp $data_dir/$filename root@9.146.172.40:$bpe_server_dir/$task
# scp root@9.146.172.40:$bpe_server_dir/$task/$filename $data_dir
# # scp root@9.146.172.40:$bpe_server_dir/$task/$filename $data_dir
# mv $data_dir/$filename $data_dir/$task"_sents"$part_tail".bpe.en"
# # ==========


# # ==========
# # Merging multiple txts into a jsonl file (e.g., input and pred output for transart)
# # ==========
# para_dir=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data
# task=trip_advisor
# data_dir=$para_dir/$task/transmart_en2zh
# out_dir=$para_dir/$task/en2zh_jsonl
# # for i in $(seq 0 3);
# # do
#   # export part_tail=-part0$i
# # done
# export part_tail=-part02
# python utils.py \
#   --data_dir $data_dir\
#   --txt_files $task"_sents"$part_tail"-fix.txt.bpe.en"::$task"_sents"$part_tail-"en2zh.txt.bpe.zh"\
#   --text_col src_en::pred_zh\
#   --out_dir $out_dir\
#   --out_filename $task"_sents"$part_tail"-en2zh.jsonl" \
#   --func_name txts_to_jsonl
# # ==========

# # ==========
# # Transform jsonl file to pure-text txt based one a specific key(s)
# # ==========
# PARA_DIR=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data
# # data_dir=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data/msmarco/en2zh_m2m100_418M
# task=cc_news
# # data_dir=$PARA_DIR/$task/en2zh_m2m100_418M
# data_dir=$PARA_DIR/$task
# # part_tail=-part03-en2zh
# # cc_news_sents_train-match.jsonl
# part_tail=_train-match
# text_col=pred_en
# python utils.py \
#   --data_dir $data_dir \
#   --jsonl_file $task"_sents"$part_tail".jsonl" \
#   --out_dir $data_dir\
#   --out_filename $task"_sents"$part_tail-$text_col".txt"\
#   --text_col $text_col\
#   --do_cleaning \
#   --func_name jsonl_to_txt_singleln
# # ==========


# # ==========
# # Transform jsonl file to txt sentence pair for embedding-based word align
# # ==========
# PARA_DIR=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data
# # data_dir=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data/msmarco/en2zh_m2m100_418M
# task=cc_news
# # data_dir=$PARA_DIR/$task/en2zh_m2m100_418M
# data_dir=$PARA_DIR/$task/word_align
# # part_tail=-part03-en2zh
# # cc_news_sents_train-match.jsonl
# # part_tail=_match_top1m
# part_tail=_nonmatch_100k
# text_col=src_en::pred_en
# python utils.py \
#   --data_dir $data_dir \
#   --jsonl_file $task"_sents"$part_tail".jsonl" \
#   --out_dir $data_dir\
#   --out_filename $task"_sents"$part_tail-$text_col".src-tgt"\
#   --text_col $text_col\
#   --func_name jsonl_to_text_align
#   # --do_cleaning \
# # ==========

# ==========
# Merging entity mask and word align
# ==========
PARA_DIR=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data
ALIGN_PRED_DIR=/apdcephfs/share_916081/timxthuang/para_checkpoints/phrase_align/embedding-align-preds
# data_dir=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data/msmarco/en2zh_m2m100_418M
task=cc_news
# data_dir=$PARA_DIR/$task/en2zh_m2m100_418M
data_dir=$PARA_DIR/$task/word_align
# part_tail=-part03-en2zh
# cc_news_sents_train-match.jsonl
jsonl_data_dir=$data_dir
# part_tail=_match_top1m
part_tail=_nonmatch_top100k
jsonl_file=$task"_sents"$part_tail".jsonl"
text_col=src_en_mapping
# txt_files=cc_news_top1m.preds
txt_files=cc_news_nonmatch_top100k.preds
out_name=$task"_sents"$part_tail"_add_map.jsonl"

python utils.py \
  --data_dir $data_dir \
  --jsonl_data_dir $jsonl_data_dir\
  --jsonl_file $jsonl_file \
  --txt_data_dir $ALIGN_PRED_DIR \
  --txt_files $txt_files\
  --out_dir $ALIGN_PRED_DIR\
  --out_filename $out_name\
  --text_col $text_col \
  --keep_col "src_en::pred_en::src_en_mask2text::"$text_col \
  --func_name alignment_merge\
  # --do_cleaning \
# ==========


# # ==========
# # Merge jsonl and txt into sentence pair (.jsonl format)
# # ==========
# # PRED_DIR=/apdcephfs/share_916081/timxthuang/bt_files/model_preds
# # DATA_DIR=/apdcephfs/share_916081/timxthuang/bt_files/mono_en
# # PARA_DIR=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data
# # Transmart_PRED_DIR=/apdcephfs/share_916081/timxthuang/transmart/dataset/bt_files
# # task=xsum
# # part_tail=part01
# # echo $part_tail
# # data_dir=$DATA_DIR/$task
# # # jsonl_data_dir=$PRED_DIR/$task/m2m100_418M/cnn_dm_sents-$part_tail-en2zh/zh-en_nobs-1
# # jsonl_data_dir=$data_dir
# # jsonl_file=$task"_sents-$part_tail.jsonl"
# # # txt_data_dir=$DATA_DIR/$task
# # txt_data_dir=$Transmart_PRED_DIR/$task
# # txt_files=$task"_sents-$part_tail-en2zh-zh2en.txt.bpe.en"
# # # xsum_sents-part00-en2zh-zh2en.txt.bpe.en
# # out_dir=$PARA_DIR/$task

# task=trip_advisor
# para_dir=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data
# jsonl_data_dir=$para_dir/$task/en_jsonl
# txt_data_dir=$para_dir/$task/transmart_zh2en
# out_dir=$para_dir/$task

# for i in $(seq 3 3);
# do
#   part_tail=-part0$i
#   echo part_tail
#   jsonl_file=$task"_sents"$part_tail".jsonl"
#   txt_files=$task"_sents"$part_tail"-en2zh-zh2en-fix.txt.bpe.en"
#   out_name=$task"_sents"$part_tail"-align.jsonl"
#   python utils.py \
#     --data_dir $data_dir \
#     --jsonl_data_dir $jsonl_data_dir\
#     --jsonl_file $jsonl_file \
#     --txt_data_dir $txt_data_dir \
#     --txt_files $txt_files\
#     --out_dir $out_dir\
#     --out_filename $out_name\
#     --text_col pred_en \
#     --func_name jsonl_txt_merge
#     # --keep_col src_en::pred_en::\
# done
# # ==========


# # ==========
# # Re-name jsonl file fields
# # ==========
# PARA_DIR=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data
# task=msmarco
# part_tail=all
# echo $part_tail
# key_mappings="text2src_en"
# # key_mappings="text2src_en::pred2pred_en"
# # key_mappings="src_en2src_en::pred_en2pred_en"
# jsonl_file=$task"_sents-$part_tail.jsonl"
# python utils.py \
#   --data_dir $PARA_DIR/$task \
#   --jsonl_data_dir $PARA_DIR/$task\
#   --jsonl_file $jsonl_file \
#   --out_dir $PARA_DIR/$task\
#   --out_filename $jsonl_file".tmp"\
#   --text_col $key_mappings \
#   --func_name jsonl_map_key\
#   --keep_col src_en::pred_en

# mv $PARA_DIR/$task/$jsonl_file".tmp" $PARA_DIR/$task/$jsonl_file
# # ==========


# # ==========
# # Hf model/dataset download and re-save
# # ==========
# cache_dir=/apdcephfs/share_916081/timxthuang/cache
# save_dir=/apdcephfs/share_916081/timxthuang/huggingface_models
# # model_names="google/pegasus-large::google/pegasus-multi_news::google/pegasus-cnn_dailymail"
# # model_names="facebook/mbart-large-50-one-to-many-mmt::facebook/mbart-large-50-many-to-one-mmt"
# model_names="facebook/mbart-large-50-many-to-many-mmt"
# python hf_load_save.py \
#   --save_dir $save_dir \
#   --cache_dir $cache_dir\
#   --model_names $model_names\
#   --func_name model_load_and_save
# # ==========

# # =========
# # Extract well-formed sents from raw text file
# # =========
# PARA_DATA_DIR=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data

# task_name=unseen_en
# data_dir=$PARA_DATA_DIR/$task_name

# # export ln_count=10

# # export data_dir=$PARA_DATA_DIR/$task_name/en_bpe
# # export file_extension=".txt.bpe.en"
# # # export file_extension=".jsonl"
# # export part_tail="-all"
# # export part_tail=""
# # export corpus_file=$task_name"_sents"$part_tail$file_extension
# # export prefix=$task_name"_sents"$part_tail

# # export src_file=$data_dir/$corpus_file
# # export out_file=$data_dir/$prefix-part
# # export nsplit=10

# # # export ln_count=$(wc -l < $src_file)

# # split -da 2  -l$(($ln_count/$nsplit)) $src_file $out_file --additional-suffix=$file_extension

# # export all_splits=""
# # for i in $(seq 0 $(($nsplit-1)));
# # do
# #   export all_splits=$all_splits";""part0"$i".jsonl"
# # done

# filename=unseen_en.txt
# export func_name=mono_sent_split_text_file
# python backtrans_process.py\
#   --data_dir $data_dir\
#   --sub_files $filename\
#   --nsplit 1\
#   --func_name $func_name\
#   --out_filename $task_name"_sents.jsonl"
# # =========

# # =========
# # Extract well-formed sents from raw text file
# # =========
# PARA_DIR=/apdcephfs/share_916081/timxthuang/paraphrase_pair_data
# task_name=unseen_en
# base_model=facebook_bart-large
# pred_path=$PARA_DIR/$task_name/facebook_bart-large-preds
# # pred_path=$PARA_DIR/$task_name/$base_model-$task_name"-preds"
# jsonl_file=unseen_en_sents.jsonl
# sort_filename=unseen_en_sents_sort.jsonl
# text_col="src_en::pred"

# func_name=backtrans_div_measure
# python utils.py \
#   --data_dir $pred_path \
#   --jsonl_file $jsonl_file \
#   --out_dir $pred_path\
#   --out_filename $sort_filename\
#   --text_col $text_col \
#   --func_name $func_name\
#   --do_sort
# # =========



