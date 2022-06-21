import sys
import os
import logging
import json
import argparse
import pandas as pd
import subprocess
import re
import spacy
import nltk
import linecache
import copy
from ftfy import fix_text
from tqdm.auto import tqdm
from pathlib import Path
from typing import Dict, Callable

logger = logging.getLogger(__name__)
logging.basicConfig(
    # filename=os.path.join(args.output_dir, args.logname),
    # filemode='w',
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
PROCESSED_DIR = "/Users/timhuang/Desktop/paraphrase_datasets/Processed"


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn
    return docstring_decorator


def count_entail(*labels):
    return sum([1 if label == "entailment" else 0 for label in labels])


def infer_label(count):
    return 1 if count > 3 else 0


def binarize(label):
    return 1 if label == "entailment" else 0


def value_binarize(value, threshold=3.0):
    if value == -1:
        return -1
    else:
        return 1 if value > threshold else 0


def tweet_major_vote(value, threshold=0.6):
    value = value.replace("(", "").replace(")", "")
    pos, total = value.split(",")
    pos, total = int(pos), int(total)
    return 1 if pos / total > threshold else 0


def file_merging(work_dir, filenames, out_filename):
    filenames = [str(os.path.join(work_dir, filename)) for filename in filenames]
    out_file = str(os.path.join(work_dir, out_filename))
    command = "cat " + " ".join(filenames) + " > " + out_file
    print(command)
    subprocess.call(command, shell=True)


def backtrans_div_measure(args):
    filename = args.jsonl_file
    print(f"{filename},")
    filename_prefix = filename.split(".")[0]
    col1, col2 = args.text_col.split("::")
    src_file = os.path.join(args.data_dir, filename)
    out_file = os.path.join(args.out_dir, args.out_filename)
    # ln_count = sum(1 for _ in Path(src_file).open(encoding="utf-8", errors='ignore').readlines())
    # progress_bar = tqdm(range(ln_count), disable=not args.major_process)
    spacy_pipe = spacy.load("en_core_web_trf")
    # spacy_pipe.add_pipe("language_detector")

    def get_div(sent1, sent2):
        # sent1 = sent1_text.strip().lower().split(" ")
        # sent2 = sent2_text.strip().lower().split(" ")
        div_score = nltk.edit_distance(sent1, sent2) / max(len(sent1), len(sent2))
        return div_score

    def not_valid(sent1, sent2):
        # sent1 = sent1_text.strip().lower().split(" ")
        # sent2 = sent2_text.strip().lower().split(" ")
        max_len, min_len = max(len(sent1), len(sent2)), min(len(sent1), len(sent2))
        if max_len > 100 or min_len < 5:
            return True
        if (max_len / min_len > 1.5 and min_len > 20) or\
                (max_len / min_len > 1.9 and min_len <= 20):
            # print(f"{sent1_text} {len(sent1)}\n{sent2_text} {len(sent2)}\n")
            return True
        if "unk" in sent1 + sent2:
            return True
        res_1 = spacy_pipe(" ".join(sent1))
        res_2 = spacy_pipe(" ".join(sent2))
        if res_1._.language != "en" or res_2._.language != "en":
            return True
        return False
    
    div_score_all = 0
    all_div = list()
    with open(src_file) as f_in:
        all_lns = f_in.readlines()
        ln_count = len(all_lns)
        progress_bar = tqdm(range(ln_count))
        for idx in range(ln_count):
            progress_bar.update(1)
            content = json.loads(all_lns[idx])
            src, pred = content[col1], content[col2]
            sent1 = src.strip().lower().split(" ")
            sent2 = pred.strip().lower().split(" ")

            # if not_valid(sent1, sent2):
            #     invalid_num += 1
            #     continue

            div_score = get_div(sent1, sent2)
            div_score_all += div_score
            content["div_score"] = div_score
            all_div.append((content, div_score))

    print(round(div_score_all / ln_count, 4))
    
    if args.do_sort:
        sorted_input_div = sorted(all_div, key=lambda x: x[1], reverse=False)
        sorted_input = [_content for (_content, _) in sorted_input_div]
    with open(out_file, "w") as f_out:
        for item in sorted_input:
            json.dump(
                item,
                f_out,
                ensure_ascii=False
            )
            f_out.write("\n")


def jsonl_to_txt(args):
    # keep_col = args.keep_col.split("::")
    # assert args.keep_col, f"keep_col must be given instead of {args.keep_col}"
    in_file, out_file = os.path.join(args.data_dir, args.jsonl_file),\
        os.path.join(args.data_dir, args.out_filename)
    with open(in_file) as f_in, open(out_file, "w") as f_out:
        all_lines = f_in.readlines()

        if args.do_sort:
            input_lens = [
                (i, len(json.loads(line)["src_en"].split())) for i, line in enumerate(all_lines)]
            sorted_input_lens = sorted(input_lens, key=lambda x: x[1],
                                       reverse=False)
            sorted_keys = {}
            sorted_inputs = []

            for i, (idx, _) in enumerate(sorted_input_lens):
                sorted_inputs.append(all_lines[idx])
                sorted_keys[idx] = i
            iter_lns = sorted_inputs 
        else:
            iter_lns = all_lines

        for json_s in iter_lns:
            content = json.loads(json_s)
            # ln_content = [content[col] for col in keep_col]
            src_text = content["src_en"]
            src_text_zh = content["src_en2zh"]
            tgt_text = content["pred_en"]
            tgt_text_zh = content["pred_en2zh"]
            ln_content = [f"{src_text} ({src_text_zh})", f"{tgt_text} ({tgt_text_zh})"]
            f_out.write("\t".join(ln_content))
            f_out.write("\n")


def jsonl_to_text_align(args):
    text_col = args.text_col.split("::")
    assert len(text_col) == 2, f"Current alignment only support sentence pair while {len(text_col)} given"
    col1, col2 = text_col
    in_file, out_file = os.path.join(args.data_dir, args.jsonl_file),\
        os.path.join(args.out_dir, args.out_filename)
    with open(in_file, encoding="utf-8") as f_in,\
            open(out_file, "w", encoding="utf-8") as f_out:
        all_ln = f_in.readlines()
        progress_bar = tqdm(range(len(all_ln)))
        for json_s in all_ln:
            content = json.loads(json_s)
            src_sent, tgt_sent = content.get(col1).strip(), content.get(col2).strip()
            if args.do_cleaning:  # todo: add pre-processing, tokenizing, etc
                pass
            f_out.write(" ||| ".join([src_sent, tgt_sent]))
            f_out.write("\n")
            progress_bar.update(1)


def jsonl_to_txt_singleln(args):
    text_col = args.text_col.split("::")
    assert args.text_col, f"text_col must be given instead of {args.text_col}"
    in_file, out_file = os.path.join(args.data_dir, args.jsonl_file),\
        os.path.join(args.out_dir, args.out_filename)

    with open(in_file, encoding="utf-8") as f_in,\
            open(out_file, "w", encoding="utf-8") as f_out:
        all_ln = f_in.readlines()
        progress_bar = tqdm(range(len(all_ln)))
        for json_s in all_ln:
            content = json.loads(json_s)
            for key in text_col:
                _text = content.get(key, "").strip()
                if args.do_cleaning:
                    _text.replace("\n", "")
                    # _text.replace("\\", "")
                    if not _text or len(_text) < 20:
                        continue
                    _text = fix_text(_text)
                # ln_content = [content[col] for col in text_col]
                f_out.write(_text)
                f_out.write("\n")
            progress_bar.update(1)


def json_to_txt_singleln(args):
    assert "trip_advisor" in args.jsonl_file, f"Only json structure of trip_advisor review data {args.jsonl_file}"
    in_file, out_file = os.path.join(args.data_dir, args.jsonl_file),\
        os.path.join(args.out_dir, args.out_filename)
    with open(in_file) as f_in, open(out_file, "w") as f_out:
        content = json.load(f_in)
        progress_bar = tqdm(range(len(content)))
        for entity in content:
            review_texts = entity["reviews"]
            for review in review_texts:
                sentences = review["sentences"]
                for sent in sentences:
                    _text = sent.strip()
                    _text = _text.replace("\n", "")
                    # _text.replace("\\", "")
                    if not _text or len(_text) < 20:
                        continue
                    # ln_content = [content[col] for col in text_col]
                    f_out.write(fix_text(_text))
                    f_out.write("\n")
            progress_bar.update(1)


def txts_to_jsonl(args):
    assert args.txt_files and args.text_col, f"txt_files and text_col MUST be given"
    txt_files = args.txt_files.split("::")
    text_col = args.text_col.split("::")
    assert len(txt_files) == len(text_col), f""
    txt_sents = list()
    ln_count = None
    for filename in txt_files:
        with open(os.path.join(args.data_dir, filename)) as f_in:
            _tmp_file_lns = f_in.readlines()
            txt_sents.append(_tmp_file_lns)
            if ln_count is None:
                ln_count = len(_tmp_file_lns)
            else:
                assert ln_count == len(_tmp_file_lns), f""
    with open(os.path.join(args.out_dir, args.out_filename), "w") as f_out:
        progress_bar = tqdm(range(ln_count))
        for idx in range(ln_count):
            sent_seq = [txt_file[idx] for txt_file in txt_sents]
            _tmp_example = dict()
            for k, v in zip(text_col, sent_seq):
                if k == "src_en":
                    v = re.sub("(@@ )|(@@ ?$)", "", v)
                elif k == "pred_zh":
                    v = v.replace(" ", "")
                _tmp_example[k] = v.strip()
            json.dump(
                _tmp_example,
                f_out,
                ensure_ascii=False
            )
            f_out.write("\n")
            progress_bar.update(1)


def alignment_merge(args):
    keep_col = args.keep_col.split("::") if args.keep_col else None
    logger.info(f"keep_col: {keep_col}")
    assert args.jsonl_file and args.txt_files and args.text_col, f"jsonl_file and txt_files MUST be given"
    jsonl_infile, txt_infile = os.path.join(args.jsonl_data_dir, args.jsonl_file),\
        os.path.join(args.txt_data_dir, args.txt_files)
    with open(jsonl_infile) as f_in1, open(txt_infile) as f_in2:
        jsonl_lns = f_in1.readlines()
        txt_lns = f_in2.readlines()
    assert len(jsonl_lns) == len(txt_lns), f"Num of lines must be equal, instead of {len(jsonl_lns)} vs {len(txt_lns)}"
    ln_count = len(jsonl_lns)
    
    VALID_TYPES = ["EVENT", "FAC", "GPE", "LAW", "LOC", "ORDINAL", "ORG", "PERSON", "WORK_OF_ART"]
    
    def _extract_all_align(json_content, alignment):
        src_tokens = json_content["src_en"].split()
        tgt_tokens = json_content["pred_en"].split()
        alignment = alignment.strip().split(" ")
        alignment_map = {int(align.split("-")[0]): int(align.split("-")[1]) for align in alignment}
        src_en_align = list()
        for idx, token in enumerate(src_tokens):
            if idx not in alignment_map:
                src_en_align.append((token, None))
            else:
                tgt_idx = alignment_map[idx]
                src_en_align.append((token, tgt_tokens[tgt_idx]))
        return src_en_align

    def _extract_entity_mapping(json_content, alignment):
        src_en_align = dict()
        src_en_mask2text = json_content["src_en_mask2text"]
        if len(src_en_mask2text) == 0:
            return src_en_align
        
        logging.debug(f"alignment: {alignment}")
        alignment = alignment.strip().split(" ")
        alignment_map = {int(align.split("-")[0]): int(align.split("-")[1]) for align in alignment}
        logger.debug(f"alignment_map: {alignment_map}")
        src_mask_tokens = json_content["src_en_mask"].split()
        tgt_tokens = json_content["pred_en"].split()

        non_mask_idx = 0
        for idx, token in enumerate(src_mask_tokens):

            if "-" not in token or token.split("-")[0] not in VALID_TYPES:
                non_mask_idx += 1
            else:
                token = f"{token.split('-')[0]}-{token.split('-')[1][0]}"
                logger.debug(f"idx, token: {idx} {token}")
                span_align_res = dict()
                # src_en_align[token] = 
                ent_text = src_en_mask2text[token]
                span_align_res["ori_span"] = ent_text
                span_align_res["word_align"] = list()
                span = ent_text.split(" ")
                for _span_idx, _span_text in enumerate(span):
                    _src_en_idx = non_mask_idx + _span_idx
                    if _src_en_idx in alignment_map:
                        _tgt_en_idx = alignment_map[_src_en_idx]
                        _tgt_en_text = tgt_tokens[_tgt_en_idx]
                    else:
                        _tgt_en_text = None
                    span_align_res["word_align"].append((_span_text, _tgt_en_text))
                    logger.debug(f"(_span_text, _tgt_en_text): {(_span_text, _tgt_en_text)}")
                non_mask_idx += len(span)
                src_en_align[token] = span_align_res
        return src_en_align
    
    with open(os.path.join(args.out_dir, args.out_filename), "w", encoding="utf-8") as f_out:
        ln_count = min(ln_count, 1000)
        progress_bar = tqdm(range(ln_count))
        for idx in range(ln_count):
            json_content = json.loads(jsonl_lns[idx])
            alignment = txt_lns[idx].strip()
            # if args.text_col != "pred_en":
            #     txt_content = txt_content.replace(" ", "")
            assert args.text_col not in json_content, f"Dumplicate text_col {args.text_col} in json keys:{json_content.keys()}"

            # src_en_mapping = _extract_entity_mapping(json_content, alignment)
            # json_content[args.text_col] = copy.copy(src_en_mapping)
            src_all_mapping = _extract_all_align(json_content, alignment)
            json_content[args.text_col] = copy.copy(src_all_mapping)
            if keep_col is not None:
                _save_dict = {key: json_content[key] for key in keep_col}
            else:
                _save_dict = json_content
            json.dump(
                _save_dict,
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")
            progress_bar.update(1)


def jsonl_txt_merge(args):
    keep_col = args.keep_col.split("::") if args.keep_col else None
    logger.info(f"keep_col: {keep_col}")
    assert args.jsonl_file and args.txt_files and args.text_col, f"jsonl_file and txt_files MUST be given"
    jsonl_infile, txt_infile = os.path.join(args.jsonl_data_dir, args.jsonl_file),\
        os.path.join(args.txt_data_dir, args.txt_files)
    with open(jsonl_infile) as f_in1, open(txt_infile) as f_in2:
        jsonl_lns = f_in1.readlines()
        txt_lns = f_in2.readlines()
    assert len(jsonl_lns) == len(txt_lns), f"Num of lines must be equal, instead of {len(jsonl_lns)} vs {len(txt_lns)}"
    ln_count = len(jsonl_lns)
    with open(os.path.join(args.out_dir, args.out_filename), "w") as f_out:
        progress_bar = tqdm(range(ln_count))
        for idx in range(ln_count):
            json_content = json.loads(jsonl_lns[idx])
            txt_content = txt_lns[idx].strip()
            if args.text_col != "pred_en":
                txt_content = txt_content.replace(" ", "")
            assert args.text_col not in json_content, f"Dumplicate text_col {args.text_col} in json keys:{json_content.keys()}"
            json_content[args.text_col] = txt_content
            if keep_col is not None:
                _save_dict = {key: json_content[key] for key in keep_col}
            else:
                _save_dict = json_content
            json.dump(
                _save_dict,
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")
            progress_bar.update(1)


def BPE_decode(args):
    assert args.txt_files and args.text_col, f"txt_files and text_col MUST be given! Here, text_col specifies the processing type"
    with open(os.path.join(args.data_dir, args.txt_files)) as f_in:
        _tmp_file_lns = f_in.readlines()
    ln_count = len(_tmp_file_lns)
    with open(os.path.join(args.out_dir, args.out_filename), "w") as f_out:
        progress_bar = tqdm(range(ln_count))
        for idx in range(ln_count):
            _text = _tmp_file_lns[idx].strip()
            if args.text_col == "src_en":
                _text = re.sub("(@@ )|(@@ ?$)", "", _text)
            else:
                raise NotImplementedError(f"Processing type {args.text_col} need to be implemented")
            f_out.write(_text)
            f_out.write("\n")
            progress_bar.update(1)
    

def jsonl_map_key(args):
    assert args.jsonl_file and args.text_col, f""

    keep_col = args.keep_col.split("::") if args.keep_col is not None else None
    former_file, restore_file = os.path.join(args.data_dir, args.jsonl_file),\
        os.path.join(args.out_dir, args.out_filename)
    key_mappings = args.text_col.split("::")
    key_mappings = [(pair.split("2")) for pair in key_mappings]
    logger.info(f"key_mappings: {key_mappings}")
    update_func = args.filter_func.split("::") if args.filter_func else None

    with open(former_file) as f_in, open(restore_file, "w") as f_out:
        for json_s in f_in.readlines():
            content = json.loads(json_s)
            # logger.info(content)
            for old_key, new_key in key_mappings:
                # assert old_key in content, f"Invalid key mapping: {old_key} not in {content.keys()}"
                content[new_key] = content.pop(old_key)
            if update_func is not None:
                for update_key, func in update_func.items():
                    content[update_key] = func(content[update_key])
            if keep_col is not None:
                content = {col: content[col] for col in keep_col}
            json.dump(content, f_out, ensure_ascii=False)
            f_out.write("\n")


func_name_mapping = {
    "jsonl_to_txt": jsonl_to_txt,
    "jsonl_to_txt_singleln": jsonl_to_txt_singleln,
    "json_to_txt_singleln": json_to_txt_singleln,
    "txts_to_jsonl": txts_to_jsonl,
    "jsonl_txt_merge": jsonl_txt_merge,
    "jsonl_map_key": jsonl_map_key,
    "BPE_decode": BPE_decode,
    "backtrans_div_measure": backtrans_div_measure,
    "jsonl_to_text_align": jsonl_to_text_align,
    "alignment_merge": alignment_merge,
    # "file_merging": file_merging,
}


def run_format_transform():
    parser = argparse.ArgumentParser(
        description="Download datasets from huggingface and save to jsonl files")
    parser.add_argument("--func_name", type=str, required=True, help="The function name to run")
    parser.add_argument("--data_dir", type=str, required=True, help="Target dataset names")
    parser.add_argument("--jsonl_data_dir", type=str, default=None, help="Target dataset names")
    parser.add_argument("--txt_data_dir", type=str, default=None, help="Target dataset names")
    parser.add_argument("--jsonl_file", type=str, default=None, help="Target dataset names")
    parser.add_argument("--txt_files", type=str, default=None, help="Input multiple txt files")
    parser.add_argument("--out_dir", type=str, required=True, help="Target dataset names")
    parser.add_argument("--out_filename", type=str, required=True, help="Target dataset names")
    parser.add_argument("--do_cleaning", action="store_true", help="If passed, do some basic text cleaning and filtering")
    parser.add_argument("--do_sort", action="store_true", help="If passed, do sorting based on length")
    parser.add_argument("--keep_col", type=str, default=None, help="Target dataset names")
    parser.add_argument("--text_col", type=str, default=None, help="Target dataset names")
    parser.add_argument("--filter_func", type=str, default=None, help="Target dataset names")
    args = parser.parse_args()

    func_name = args.func_name
    func_name_mapping[func_name](args)


if __name__ == '__main__':
    run_format_transform()
