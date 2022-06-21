import sys
import os
import argparse
import logging
import json
import linecache
import spacy
import pandas as pd
import editdistance
import numpy as np
from copy import copy
from pathlib import Path
from pprint import pprint
from typing import Dict, Callable
from collections import defaultdict
from statsmodels.stats import inter_rater as irr
from collections import Counter

ANNOTATED_DIR = "/Users/timhuang/Desktop/paraphrase_datasets/Annotated/evaluation_en"
en_filenames = ["evaluation_en_fangchi.txt", "evaluation_en_penny.txt", "evaluation_en_unicorn.txt"]
splits = ["coco", "flickr", "mnli", "msrvtt", "quora", "snli", "stsb", "tweets", "wikianswers"]
id_to_label = {
    1: "neg",
    2: "neutral",
    3: "pos",
}

special_ch = [".", ",", ";", "!", ":", "\\", "/", "@", ]


def inter_agreement():
    anno_file1, anno_file2, anno_file3 = [os.path.join(ANNOTATED_DIR, filename) for filename in en_filenames]
    ln_count = sum(1 for _ in Path(anno_file1).open(encoding="utf8", errors='ignore').readlines())
    rates = [[0] * 3 for _ in range(ln_count)]
    with open(anno_file1, encoding="utf8", errors='ignore') as f_in1,\
            open(anno_file2, encoding="utf8", errors='ignore') as f_in2,\
            open(anno_file3, encoding="utf8", errors='ignore') as f_in3:
        anno1_lines = f_in1.readlines()
        anno2_lines = f_in2.readlines()
        anno3_lines = f_in3.readlines()

    def get_label(line, file_idx=None):
        elements = line.rstrip("\n").split("\t")
        elements = list(filter(len, elements))  # filter out empty strings
        assert elements[-1] in ["1", "2", "3"], f"Invalid label from line: {line} in file {file_idx}"
        return int(elements[-1])

    for idx in range(ln_count):
        anno1, anno2, anno3 = get_label(anno1_lines[idx], file_idx="fangchi"),\
            get_label(anno2_lines[idx], file_idx="penny"), get_label(anno3_lines[idx], file_idx="unicorn")
        # print(f"anno1: {anno1}, anno2: {anno2}, anno3: {anno3}")
        anno1, anno3, anno3 = int(anno1), int(anno2), int(anno3)
        for anno in [anno1, anno3, anno3]:
            rates[idx][anno - 1] += 1

    default_stat = {
        "total_disagree": 0,
        "pos": 0,
        "neutral": 0,
        "neg": 0,
    }
    split_stats = dict()
    for split_name in splits:
        split_stats[split_name] = copy(default_stat)
    # overall kappa
    overall_kappa = irr.fleiss_kappa(rates, method='fleiss')
    print(f"overall_kappa: {overall_kappa}")
    # per-split kappa ans stats
    for idx, split_name in enumerate(splits):
        current_stats = split_stats[split_name]
        start_idx, end_idx = idx * 200, (idx + 1) * 200 - 1
        split_rates = rates[start_idx: end_idx + 1]
        split_kappa = irr.fleiss_kappa(split_rates, method='fleiss')
        print(f"split: {split_name} -- kappa: {split_kappa}")

        # total diverse and pos/neg based on major
        for rate in split_rates:
            if all(rate):
                current_stats["total_disagree"] += 1
            else:
                major_rate = np.argmax(rate)
                current_stats[id_to_label[major_rate + 1]] += 1
    pprint(split_stats)


def transform_to_binary():
    out_filename = "evaluation_en_voting.jsonl"
    anno_file1, anno_file2, anno_file3 = [os.path.join(ANNOTATED_DIR, filename) for filename in en_filenames]
    ln_count = sum(1 for _ in Path(anno_file2).open(encoding="utf8", errors='ignore').readlines())
    print(f"ln_count: {ln_count}")
    rates = [[0] * 3 for _ in range(ln_count)]
    with open(anno_file1, encoding="utf8", errors='ignore') as f_in1,\
            open(anno_file2, encoding="utf8", errors='ignore') as f_in2,\
            open(anno_file3, encoding="utf8", errors='ignore') as f_in3:
        anno1_lines = f_in1.readlines()
        anno2_lines = f_in2.readlines()
        anno3_lines = f_in3.readlines()

    def get_label(line, file_idx=None):
        elements = line.rstrip("\n").split("\t")
        elements = list(filter(len, elements))  # filter out empty strings
        assert elements[-1] in ["1", "2", "3"], f"Invalid label from line: {line} in file {file_idx}"
        return int(elements[-1])

    def get_content(line, idx):
        elements = line.rstrip("\n").split("\t")
        elements = list(filter(len, elements))
        # assert len(elements) == 4, f"Invalid parsing results: {elements}; of line {line}"
        if len(elements) != 4:
            print(f"Invalid parsing results: {elements}; of line {idx}")
            return None
        return {
            "sent1": elements[1],
            "sent2": elements[2],
        }

    def binarize(rate):
        return 1 if rate == 2 else 0

    with open(os.path.join(ANNOTATED_DIR, out_filename) , "w") as f_out:
        for idx in range(ln_count):
            anno1, anno2, anno3 = get_label(anno1_lines[idx], file_idx="fangchi"),\
                get_label(anno2_lines[idx], file_idx="penny"), get_label(anno3_lines[idx], file_idx="unicorn")
            # print(f"anno1: {anno1}, anno2: {anno2}, anno3: {anno3}")
            anno1, anno3, anno3 = int(anno1), int(anno2), int(anno3)
            for anno in [anno1, anno3, anno3]:
                rates[idx][anno - 1] += 1
            major_rate = np.argmax(rates[idx])
            label = binarize(major_rate)
            example = get_content(anno2_lines[idx], idx)
            if example is None:
                continue
            example["label"] = label
            json.dump(
                example,
                f_out
            )
            f_out.write("\n")


def vote_to_score():
    out_filename = "evaluation_en_score.jsonl"
    anno_file1, anno_file2, anno_file3 = [os.path.join(ANNOTATED_DIR, filename) for filename in en_filenames]
    ln_count = sum(1 for _ in Path(anno_file1).open(encoding="utf8", errors='ignore').readlines())
    rates = [[0] * 3 for _ in range(ln_count)]
    with open(anno_file1, encoding="utf8", errors='ignore') as f_in1,\
            open(anno_file2, encoding="utf8", errors='ignore') as f_in2,\
            open(anno_file3, encoding="utf8", errors='ignore') as f_in3:
        anno1_lines = f_in1.readlines()
        anno2_lines = f_in2.readlines()
        anno3_lines = f_in3.readlines()

    def get_label(line, file_idx=None):
        elements = line.rstrip("\n").split("\t")
        elements = list(filter(len, elements))  # filter out empty strings
        assert elements[-1] in ["1", "2", "3"], f"Invalid label from line: {line} in file {file_idx}"
        return int(elements[-1])

    def get_content(line, idx):
        elements = line.rstrip("\n").split("\t")
        elements = list(filter(len, elements))
        # assert len(elements) == 4, f"Invalid parsing results: {elements}; of line {line}"
        if len(elements) != 4:
            print(f"Invalid parsing results: {elements}; of line {idx}")
            return None
        return {
            "sent1": elements[1],
            "sent2": elements[2],
        }

    def get_score(anno1, anno2, anno3):
        return (anno1 + anno2 + anno3) / 9.0

    with open(os.path.join(ANNOTATED_DIR, out_filename) , "w") as f_out:
        for idx in range(ln_count):
            anno1, anno2, anno3 = get_label(anno1_lines[idx], file_idx="fangchi"),\
                get_label(anno2_lines[idx], file_idx="penny"), get_label(anno3_lines[idx], file_idx="unicorn")
            example = get_content(anno2_lines[idx], idx)
            score = get_score(anno1, anno2, anno3)
            example["score"] = str(round(score, 2))
            json.dump(
                example,
                f_out
            )
            f_out.write("\n")


def get_filter_stats():
    pred_dir = "/Users/timhuang/Desktop/paraphrase_datasets/Annotated/preds/"
    preds_file = pred_dir + "evaluation_en_voting-preds.jsonl"
    score_file = ANNOTATED_DIR + "/evaluation_en_score.jsonl"
    ln_count = sum(1 for _ in Path(preds_file).open(encoding="utf8", errors='ignore').readlines())
    with open(preds_file, encoding="utf8", errors='ignore') as f_in1,\
            open(score_file, encoding="utf8", errors='ignore') as f_in2:
        preds_lines = f_in1.readlines()
        score_lines = f_in2.readlines()

    default_stat = {
        "pos_num": 0,
        "pre_relevance": 0,
        "post_human_relevance": 0,
        "post_model_relevance": 0,
        "pre_diverse": 0,
        "post_human_diverse": 0,
        "post_model_diverse": 0,
    }
    instance_record = {"rel_score": 0, "div_score": 0, "label": None, "pred": None, }
    stats = [copy(instance_record) for _ in range(ln_count)]
    split_stats = dict()

    def get_rel(line):
        content = json.loads(line)
        return float(content["score"])

    def get_div(line):
        content = json.loads(line)
        sent1, sent2 = content["sent1"], content["sent2"]
        sent1 = sent1.strip().lower().split(" ")
        sent2 = sent2.strip().lower().split(" ")
        for ch in special_ch:
            sent1.remove(ch) if ch in sent1 else None
            sent2.remove(ch) if ch in sent2 else None
        div_score = editdistance.eval(sent1, sent2) / max(len(sent1), len(sent2))
        if div_score > 1.0:
            print(div_score, sent1, sent2)
        return div_score

    for split_name in splits:
        split_stats[split_name] = copy(default_stat)
    
    # for idx in range(200):
    for idx in range(ln_count):
        line = preds_lines[idx]
        content = json.loads(line)
        label, pred = content["label"], content["pred"]
        rel_score = get_rel(score_lines[idx])
        div_score = get_div(line)
        stats[idx]["rel_score"] = rel_score
        stats[idx]["div_score"] = div_score
        stats[idx]["label"], stats[idx]["pred"] = label, pred
        # print(rel_score, div_score, label, pred)

    all_pos = 0
    all_pre_relevance = 0
    all_post_relevance = 0
    all_pre_diverse = 0
    all_post_diverse = 0
    all_human_rel = 0
    all_human_div = 0
    for idx, split_name in enumerate(splits):
        current_stats = split_stats[split_name]
        start_idx, end_idx = idx * 200, (idx + 1) * 200 - 1
        split_records = stats[start_idx: end_idx + 1]
        # pprint(split_records)
        hit = 0
        pos_label = 0
        pos_num = 0
        pre_relevance = 0
        pre_diverse = 0
        post_model_relevance = 0
        post_model_diverse = 0
        post_human_relevance = 0
        post_human_diverse = 0
        for record in split_records:
            pre_relevance += record["rel_score"]
            pre_diverse += record["div_score"]
            label, pred = record["label"], record["pred"]
            if pred == 1:
                pos_num += 1
                post_model_relevance += record["rel_score"]
                post_model_diverse += record["div_score"]
            if label == 1:
                pos_label += 1
                post_human_relevance += record["rel_score"]
                post_human_diverse += record["div_score"]
            if pred == 1 and label == 1:
                hit += 1
        avg_pre_relevance = round(pre_relevance / 200, 2) 
        avg_pre_diverse = round(pre_diverse / 200, 2) 
        avg_post_model_relevance = round(post_model_relevance / pos_num, 2) 
        avg_post_model_diverse = round(post_model_diverse / pos_num, 2) 
        avg_post_human_relevance = round(post_human_relevance / pos_label, 2) 
        avg_post_human_diverse = round(post_human_diverse / pos_label, 2) 
        all_pos += pos_num
        all_pre_relevance += avg_pre_relevance
        all_post_relevance += avg_post_model_relevance
        all_pre_diverse += avg_pre_diverse
        all_post_diverse += avg_post_model_diverse
        all_human_rel += avg_post_human_relevance
        all_human_div += avg_post_human_diverse
        print(f"split: {split_name}: pos_num {pos_num} hit {hit}|| relevance {avg_pre_relevance} VS {avg_post_model_relevance} ||\
            diverse:{avg_pre_diverse} VS {avg_post_model_diverse}")
        print(f"split: {split_name} relevance {avg_pre_relevance} VS {avg_post_human_relevance} || \
            diverse:{avg_pre_diverse} VS {avg_post_human_diverse}")    
        print("\n")
    print(f"ALL: pos_num {all_pos} || relevance {all_pre_relevance} VS {all_post_relevance} ||\
            diverse:{all_pre_diverse} VS {all_post_diverse}")
    print(f"ALL: relevance {all_pre_relevance} VS {all_human_rel} || \
        diverse:{all_pre_diverse} VS {all_human_div}")    


def measure_div():
    pred_dir = "/Users/timhuang/Desktop/"
    file1, file2 = pred_dir + "msmarco_sents-10k-preds-zh-en-preds.jsonl",\
        pred_dir + "msmarco_sents-10k-en-zh-preds.jsonl"
    
    with open(file1, encoding="utf8") as f_in1, open(file2, encoding="utf8") as f_in2:
        preds_f1 = f_in1.readlines()
        preds_f2 = f_in2.readlines()

    ln_count = sum(1 for _ in Path(file1).open(encoding="utf8", errors='ignore').readlines())
    print(f"ln_count: {ln_count}")

    def get_div(sent1_text, sent2_text):
        sent1 = sent1_text.strip().lower().split(" ")
        sent2 = sent2_text.strip().lower().split(" ")
        for ch in special_ch:
            sent1.remove(ch) if ch in sent1 else None
            sent2.remove(ch) if ch in sent2 else None
        div_score = editdistance.eval(sent1, sent2) / max(len(sent1), len(sent2))
        if div_score < 0.4:
            print(div_score, sent1, sent2)

        # print(div_score, sent1_text, sent2_text)

        return div_score
    
    div_score_all = 0
    for idx in range(ln_count):
        line1 = preds_f1[idx]
        line2 = preds_f2[idx]
        src_text, tgt_text = json.loads(line1)["pred"], json.loads(line1)["text"]
        div_score = get_div(src_text, tgt_text)
        div_score_all += div_score
    print(round(div_score_all / ln_count, 4))


def measure_back_div():
    parser = argparse.ArgumentParser(
        description="Removing empty pred lines")
    parser.add_argument("--data_dir", type=str, required=True, help="Target processing file dir")
    parser.add_argument("--sub_files", type=str, required=True, help="Names of the split files, seperated by ';'")
    parser.add_argument("--out_files", type=str, default=None, help="Names of the filter out files, seperated by ';'")
    parser.add_argument("--threshold", type=float, default=0.4, help="Names of the filter out files, seperated by ';'")
    args = parser.parse_args()
    # sub_files = args.sub_files
    sub_files = args.sub_files.split("::")
    out_files = None
    if args.out_files is not None:
        out_files = args.out_files.split("::")
        assert len(sub_files) == len(out_files), f"{len(sub_files)} and {len(out_files)} should be the same"
    print(f"{sub_files} {out_files}")

    # spacy_pipe = spacy.load("en_core_web_trf")

    def get_div(sent1_text, sent2_text):
        sent1 = sent1_text.strip().lower().split(" ")
        sent2 = sent2_text.strip().lower().split(" ")
        # doc1 = spacy_pipe(sent1)
        for ch in special_ch:
            sent1.remove(ch) if ch in sent1 else None
            sent2.remove(ch) if ch in sent2 else None
        div_score = editdistance.eval(sent1, sent2) / max(len(sent1), len(sent2))
        # if div_score > 0.7:
        #     print(f"{sent1_text}\n{sent2_text}\n")
        return div_score

    def heuristic_filter(sent1_text, sent2_text):
        sent1 = sent1_text.strip().lower().split(" ")
        sent2 = sent2_text.strip().lower().split(" ")
        for ch in special_ch:
            sent1.remove(ch) if ch in sent1 else None
            sent2.remove(ch) if ch in sent2 else None
        max_len, min_len = max(len(sent1), len(sent2)), min(len(sent1), len(sent2))
        if max_len > 50 or min_len < 7:
            return False
        if (max_len / min_len > 1.4 and min_len > 20) or\
                (max_len / min_len > 1.8 and min_len <= 20):
            print(f"{sent1_text} {len(sent1)}\n{sent2_text} {len(sent2)}\n")
            return False
        return True

    high_edit_res = list()
    for f_idx, file in enumerate(sub_files):
        all_div = 0
        with open(os.path.join(args.data_dir, file), encoding="utf8") as f_in:
            pred_lns = f_in.readlines()
            for pred in pred_lns:
                content = json.loads(pred)

                src_text = content["text"]
                tgt_text = content["pred"]
                _div_score = get_div(src_text, tgt_text)
                all_div += _div_score
                if _div_score >= args.threshold and heuristic_filter(src_text, tgt_text):
                    high_edit_res.append(pred)
            avg_div = all_div / len(pred_lns)
            print(f"{file} avg-div: {round(avg_div, 3)}")
        if out_files is not None:
            with open(os.path.join(args.data_dir, out_files[f_idx]), "w") as f_out:
                for pred in high_edit_res:
                    f_out.write(pred)


if __name__ == '__main__':
    # inter_agreement()
    # transform_to_binary()
    # vote_to_score()
    # get_filter_stats()
    # measure_div()
    measure_back_div()
