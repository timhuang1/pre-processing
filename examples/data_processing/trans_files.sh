PRED_DIR=/apdcephfs/share_916081/timxthuang/bt_files/model_preds
DATA_DIR=/apdcephfs/share_916081/timxthuang/bt_files/mono_en
transmart_data_path=/apdcephfs/share_916081/timxthuang/transmart/dataset/bt_files
processing_script_dir=/apdcephfs/share_916081/timxthuang/cond_gen/examples/data_processing

export task_name=xsum
export bpe_server_dir=/data1/tokenize/bpe_files

# # =====================
# # for upload mono_en txt files
# # =====================
# export prefix=$task_name"_sents"
# for i in $(seq 0 3);
# do
#   scp $PRED_DIR/$task_name/$prefix-part0$i"-en2zh.txt" root@9.146.172.40:$bpe_server_dir/$task_name
# done
# # =====================

# =====================
# for download mono_en txt files (too)
# =====================
export prefix=$task_name"_sents"
for i in $(seq 0 3);
do
  scp root@9.146.172.40:$bpe_server_dir/$task_name/$prefix-part0$i"-en2zh.txt.token.bpe" $transmart_data_path/$task_name/
  mv $transmart_data_path/$task_name/$prefix-part0$i"-en2zh.txt.token.bpe" $transmart_data_path/$task_name/$prefix-part0$i"-en2zh.txt.bpe.zh"
done
# =====================



export model_name=m2m100_418M
export src_lang_code=en
export tgt_lang_code=zh
export type_path=xsum_sents
# export part_tail=-part03
export type_path=$type_path$part_tail
export file_extension=.jsonl
export timestamp="07-06-2022_15-26"
export n_obs="50000"

export pred_path=$PRED_DIR/$task_name/$type_path/$model_name/$src_lang_code-$tgt_lang_code"_nobs"$n_obs"_"$timestamp
export pred_filename=$type_path-$src_lang_code"2"$tgt_lang_code

# # =====================
# # for upload pred result
# python $processing_script_dir/utils.py\
#   --data_dir $pred_path\
#   --jsonl_file $pred_filename".jsonl"\
#   --out_dir $pred_path\
#   --out_filename $pred_filename".txt"\
#   --text_col pred\
#   --func_name jsonl_to_txt_singleln\

# scp $pred_path/$pred_filename".txt" root@9.146.172.40:$bpe_server_dir
# # =====================

# # =====================
# # for download pred bpe files back
# export post_bpe_file=$pred_filename".txt.token.bpe"
# export re_name=$pred_filename".txt.bpe."$tgt_lang_code
# export transmart_path=/apdcephfs/share_916081/timxthuang/transmart/dataset/bt_files/$task_name
# scp root@9.146.172.40:$bpe_server_dir/$post_bpe_file $transmart_path
# mv $transmart_path/$post_bpe_file  $transmart_path/$re_name
# # =====================