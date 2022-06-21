import sys
import os
import logging
import json
import random
import linecache
import editdistance
import itertools
import pandas as pd
from typing import Dict, Callable
from pathlib import Path
from collections import defaultdict

from file_linking import DATASET_MAP, PROCESSED_DATASET_MAP
from utils import count_entail, infer_label, binarize, value_binarize, add_start_docstrings
from utils import tweet_major_vote

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

special_ch = [".", ",", ";", "!", ":", "\\", "/", "@",]

RAW_DIR = "/Users/timhuang/Desktop/paraphrase_datasets"
PROCESSED_DIR = "/Users/timhuang/Desktop/paraphrase_datasets/Processed"
SAMPLED_DIR = "/Users/timhuang/Desktop/paraphrase_datasets/Processed/samples"
POS_DIR = "/Users/timhuang/Desktop/paraphrase_datasets/Processed/valid_pos"

coco = "coco/train.jsonl"
flickr = "flickr/train.jsonl"
mnli = "mnli/val.jsonl"
snli = "snli/val.jsonl"
stsb = "stsb/validation.jsonl"
tweets = "tweets/train.jsonl"
msrvtt = "msrvtt/train.jsonl"
quora = "quora/train.jsonl"
wikianswer = "wikianswers/WikiAnswers_pairs_5m.jsonl"

sampling_task = {
    "pair": {
        "mnli": mnli,
        "snli": snli,
        "stsb": stsb,
        "quora": quora,
        "tweets": tweets,
    },
    "cluster": {
        "coco": coco,
        "flickr": flickr,
        "msrvtt": msrvtt,
    }
}

mnli_train = "mnli/train.jsonl"
snli_train = "snli/train.jsonl"
stsb_train = "stsb/train.jsonl"
quora_train = "quora/train.jsonl"
quora_test = "quora/test.jsonl"
tweets_train = "tweets/train.jsonl"

pos_task = {
    "pair": {
        "mnli": mnli_train,
        "snli": snli_train,
        "stsb": stsb_train,
        "quora": quora_train,
        "tweets": tweets_train,
    },
    "cluster": {
        "coco": coco,
        "flickr": flickr,
        "msrvtt": msrvtt,
    },
    "unlabel": {
        "wikianswer": wikianswer,
        "quora_test": quora_test,
    },
}


def sample_from_unlabel(task, filename, tgt_num=None, out_dir=POS_DIR):
    max_num = 500000
    file = os.path.join(PROCESSED_DIR, filename)
    save_file = os.path.join(out_dir, task + ".txt")
    save_jsonl_file = os.path.join(out_dir, task + ".jsonl")
    pos_idx = list(range(max_num))
    random.shuffle(pos_idx)
    sentence_pool = set()
    sample_cnt = 0
    with open(file) as f_in:
        print(f"Open file succes")
        print(f"samples: {f_in.readlines(10)}")

    with open(save_file, "w") as f_out, open(save_jsonl_file, "w") as f_out2:
        for idx in pos_idx:
            idx += 1  # linecache start at one
            line = linecache.getline(str(file), idx).rstrip("\n")
            if not line:
                logger.info(f"empty line {line} at {idx}")
                continue
            sample = json.loads(line)
            if task == "wikianswer":
                sent1, sent2 = sample
            else:
                sent1, sent2 = sample["sent1"], sample["sent2"]
            # logger.info(f"pair: {sent1}\n{sent1}")
            if pair_filter(sent1, sent2)\
                    and sent1 not in sentence_pool and sent2 not in sentence_pool:
                sentence_pool.add(sent1)
                sentence_pool.add(sent2)
                # logger.info(f"pair {idx} added")
                f_out.write(f"{task}-{sample_cnt}" + "\t" + sent1 + "\t" + sent2)
                f_out.write("\n")
                example = {
                    "idx": f"{task}-{sample_cnt}",
                    "sent1": sent1,
                    "sent2": sent2, 
                    "label": 1,
                }
                json.dump(
                    example,
                    f_out2
                )
                f_out2.write("\n")
                sample_cnt += 1
            if tgt_num is not None and sample_cnt == tgt_num: break
        logger.info(f"{task} have valid {sample_cnt} positive pairs")


def pair_filter(sent1, sent2):
    sent1 = sent1.strip().lower().split(" ")
    sent2 = sent2.strip().lower().split(" ")
    for ch in special_ch:
        sent1.remove(ch) if ch in sent1 else None
        sent2.remove(ch) if ch in sent2 else None
    min_len, max_len = min(len(sent1), len(sent2)), max(len(sent1), len(sent2))
    if min_len < 10: return False
    if min_len / max_len < 0.5: return False
    edits = editdistance.eval(sent1, sent2)
    if edits < 4: return False
    # if edits > 10: return False
    return True


def sample_from_pairs(task, filename, tgt_num=None, out_dir=SAMPLED_DIR):
    file = os.path.join(PROCESSED_DIR, filename)
    save_file = os.path.join(out_dir, task + ".txt")
    save_jsonl_file = os.path.join(out_dir, task + ".jsonl")
    pos_idx = list()
    for idx, line in enumerate(Path(file).open().readlines()):
        sample = json.loads(line)
        if sample["label"] == 1: pos_idx.append(idx)
    logger.info(f"{task} have {len(pos_idx)} positive pairs")
    random.shuffle(pos_idx)
    sample_cnt = 0
    with open(save_file, "w") as f_out, open(save_jsonl_file, "w") as f_out2:
        for idx in pos_idx:
            idx += 1  # linecache start at one
            line = linecache.getline(str(file), idx).rstrip("\n")
            sample = json.loads(line)
            sent1, sent2 = sample["sent1"], sample["sent2"]
            # filter requirement
            if pair_filter(sent1, sent2):
                f_out.write(f"{task}-{sample_cnt}" + "\t" + sent1 + "\t" + sent2)
                f_out.write("\n")
                example = {
                    "idx": f"{task}-{sample_cnt}",
                    "sent1": sent1,
                    "sent2": sent2, 
                    "label": 1,
                }
                json.dump(
                    example,
                    f_out2
                )
                f_out2.write("\n")
                sample_cnt += 1
                if tgt_num is not None and sample_cnt == tgt_num: break
        logger.info(f"{task} have valid {sample_cnt} positive pairs")


def sample_from_cluster(task, filename, tgt_num=None, out_dir=SAMPLED_DIR):
    file = os.path.join(PROCESSED_DIR, filename)
    save_file = os.path.join(out_dir, task + ".txt")
    save_jsonl_file = os.path.join(out_dir, task + ".jsonl")
    ln_count = sum(1 for _ in Path(file).open().readlines())
    pos_idx = list(range(ln_count))
    logger.info(f"{task} have {len(pos_idx)} positive clusters")
    random.shuffle(pos_idx)
    sentence_pool = set()
    # sample a cluster, pick sentence pair with desired edit distance and length 
    sample_cnt = 0
    with open(save_file, "w") as f_out, open(save_jsonl_file, "w") as f_out2:
        for idx in pos_idx:
            idx += 1  # linecache start at one
            line = linecache.getline(str(file), idx).rstrip("\n")
            sample = json.loads(line)
            sentences = sample["captions"]
            for sent1, sent2 in itertools.combinations(sentences, r=2):
                if pair_filter(sent1, sent2)\
                        and sent1 not in sentence_pool and sent2 not in sentence_pool:
                    sentence_pool.add(sent1)
                    sentence_pool.add(sent2)
                    f_out.write(f"{task}-{sample_cnt}" + "\t" + sent1 + "\t" + sent2)
                    f_out.write("\n")
                    example = {
                        "idx": f"{task}-{sample_cnt}",
                        "sent1": sent1,
                        "sent2": sent2, 
                        "label": 1,
                    }
                    json.dump(
                        example,
                        f_out2
                    )
                    f_out2.write("\n")                    
                    sample_cnt += 1
                    # break
            if tgt_num is not None and sample_cnt == tgt_num: break
        logger.info(f"{task} have valid {sample_cnt} positive pairs")


def sample_test():
    for task, filename in sampling_task["pair"].items():
        sample_from_pairs(task, filename, tgt_num=200)

    for task, filename in sampling_task["cluster"].items():
        sample_from_cluster(task, filename, tgt_num=200)


def all_pos():
    # sample_from_pairs("mnli", pos_task["pair"]["mnli"], out_dir=POS_DIR)
    for task, filename in pos_task["pair"].items():
        sample_from_pairs(task, filename, out_dir=POS_DIR)

    for task, filename in pos_task["cluster"].items():
        sample_from_cluster(task, filename, out_dir=POS_DIR)

    for task, filename in pos_task["unlabel"].items():
        sample_from_unlabel(task, filename, out_dir=POS_DIR)


if __name__ == '__main__':
    # sample_test()
    all_pos()
