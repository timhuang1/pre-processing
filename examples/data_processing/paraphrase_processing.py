import sys
import os
import logging
import json
import pandas as pd
from typing import Dict, Callable
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


def jsonl_map_key(
    former_file,
    restore_file,
    key_mappings: Dict[str, str],
    update_func: Dict[str, Callable] = None,
):
    with open(former_file) as f_in, open(restore_file, "w") as f_out:
        for json_s in f_in.readlines():
            content = json.loads(json_s)
            # logger.info(content)
            for old_key, new_key in key_mappings.items():
                assert old_key in content, f"Invalid key mapping: {old_key} not in {content.keys()}"
                content[new_key] = content.pop(old_key)
            if update_func is not None:
                for update_key, func in update_func.items():
                    content[update_key] = func(content[update_key])
            json.dump(content, f_out)
            f_out.write("\n")


def remap_mnli_test():
    former_file = "/Users/timhuang/Desktop/paraphrase_datasets/NLI/multinli_1.0/test.jsonl"
    restore_file = "/Users/timhuang/Desktop/paraphrase_datasets/Processed/test.jsonl"
    key_mappings = {"premise": "sent1", "hypothesis": "sent2"}
    jsonl_map_key(former_file, restore_file, key_mappings)


@add_start_docstrings("infer label, standard for mnli:\
    Count 'entail' label from all five annotators\
    4,5 is pos and vice versa")
def process_mnli():
    task = "mnli"
    raw_files = DATASET_MAP[task]
    all_splits = list(raw_files.keys())
    sent1_key, sent2_key = "sentence1", "sentence2"
    columns = ['gold_label', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse',
               'sentence2_parse', 'sentence1', 'sentence2', 'promptID', 'pairID', 'genre', 'label1',
               'label2', 'label3', 'label4', 'label5']

    for split in all_splits:
        out_path = PROCESSED_DATASET_MAP[task][split]
        file = raw_files[split]
        logger.info(f"Read from file {file}")
        raw_df = pd.read_csv(file, sep="\t", names=columns, skiprows=1)

        # # scripts for multiple annotation available
        # raw_df["entail_count"] = raw_df.apply(
        #     lambda x: count_entail(x["label1"], x["label2"], x["label3"], x["label4"], x["label5"]),
        #     axis=1
        # )
        # if "label" in raw_df: raw_df = raw_df.drop(["label"], axis=1)
        # raw_df["label"] = raw_df.apply(
        #     lambda x: infer_label(x["entail_count"]), axis=1)
        
        raw_df["label"] = raw_df.apply(lambda x: binarize(x["gold_label"]), axis=1)
        # select and rename column
        raw_df = raw_df.rename(columns={sent1_key: "sent1", sent2_key: "sent2"})
        raw_df = raw_df[["sent1", "sent2", "label"]]
        raw_df = raw_df[raw_df['sent1'].notna()]
        raw_df = raw_df[raw_df['sent2'].notna()]
        nan_df = raw_df[raw_df.isna().any(axis=1)]
        logger.info(f"rows having null: {nan_df}")
        logger.info(f"Saving to {out_path}")
        raw_df.to_json(path_or_buf=out_path, orient='records', lines=True) 


def process_nli():
    task = "snli"
    raw_files = DATASET_MAP[task]
    all_splits = list(raw_files.keys())
    sent1_key, sent2_key = "sentence1", "sentence2"
    columns = ['gold_label', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse',
               'sentence1', 'sentence2', 'captionID', 'pairID', 'label1', 'label2', 'label3', 'label4', 'label5']
    for split in all_splits:
        out_path = PROCESSED_DATASET_MAP[task][split]
        file = raw_files[split]
        logger.info(f"Read from file {file}")
        raw_df = pd.read_csv(file, sep="\t", names=columns, skiprows=1)

        # # scripts for multiple annotation available
        # raw_df["entail_count"] = raw_df.apply(
        #     lambda x: count_entail(x["label1"], x["label2"], x["label3"], x["label4"], x["label5"]),
        #     axis=1
        # )
        # if "label" in raw_df: raw_df = raw_df.drop(["label"], axis=1)
        # raw_df["label"] = raw_df.apply(
        #     lambda x: infer_label(x["entail_count"]), axis=1)
        
        raw_df["label"] = raw_df.apply(lambda x: binarize(x["gold_label"]), axis=1)
        # select and rename column
        raw_df = raw_df.rename(columns={sent1_key: "sent1", sent2_key: "sent2"})
        raw_df = raw_df[["sent1", "sent2", "label"]]
        raw_df = raw_df[raw_df['sent1'].notna()]
        raw_df = raw_df[raw_df['sent2'].notna()]
        nan_df = raw_df[raw_df.isna().any(axis=1)]
        logger.info(f"rows having null: {nan_df}")
        logger.info(f"Saving to {out_path}")
        raw_df.to_json(path_or_buf=out_path, orient='records', lines=True) 


def process_quora():
    task = "quora"
    columns = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']

    raw_files = DATASET_MAP[task]
    all_splits = list(raw_files.keys())
    sent1_key, sent2_key = "question1", "question2"
    for split in all_splits:
        out_path = PROCESSED_DATASET_MAP[task][split]
        file = raw_files[split]
        logger.info(f"Read from file {file}")
        if split == "test":
            columns = ["test_id", "question1", "question2"]
        raw_df = pd.read_csv(file, sep=",", names=columns, skiprows=1)

        raw_df["label"] = raw_df.apply(
            lambda x: int(x["is_duplicate"]) if "is_duplicate" in raw_df else -1, axis=1)

        # select and rename column
        raw_df = raw_df.rename(columns={sent1_key: "sent1", sent2_key: "sent2"})
        raw_df = raw_df[["sent1", "sent2", "label"]]
        raw_df = raw_df[raw_df['sent1'].notna()]
        raw_df = raw_df[raw_df['sent2'].notna()]
        nan_df = raw_df[raw_df.isna().any(axis=1)]
        logger.info(f"rows having null: {nan_df}")
        logger.info(f"Saving to {out_path}")
        raw_df.to_json(path_or_buf=out_path, orient='records', lines=True)


def remap_stsb():
    former_train = "/Users/timhuang/Desktop/paraphrase_datasets/stsb/train.jsonl"
    former_val = "/Users/timhuang/Desktop/paraphrase_datasets/stsb/validation.jsonl"
    former_test = "/Users/timhuang/Desktop/paraphrase_datasets/stsb/test.jsonl"
    restore_train = "/Users/timhuang/Desktop/paraphrase_datasets/Processed/stsb/train.jsonl"
    restore_val = "/Users/timhuang/Desktop/paraphrase_datasets/Processed/stsb/validation.jsonl"
    restore_test = "/Users/timhuang/Desktop/paraphrase_datasets/Processed/stsb/test.jsonl"
    key_mappings = {"sentence1": "sent1", "sentence2": "sent2"}
    label_update = {"label": value_binarize}
    for former_file, restore_file\
            in zip([former_train, former_val, former_test], [restore_train, restore_val, restore_test]):
        jsonl_map_key(former_file, restore_file, key_mappings, update_func=label_update)


@add_start_docstrings("Currently directly use line index to fetch a list of question pair")
def process_wikianswer():
    pass


def process_twitter():
    task = "tweets"
    columns = ['sent1', 'sent2', 'annotation', 'url']
    raw_files = DATASET_MAP[task]
    all_splits = list(raw_files.keys())
    for split in all_splits:
        out_path = PROCESSED_DATASET_MAP[task][split]
        file = raw_files[split]
        logger.info(f"Read from file {file}")
        raw_df = pd.read_csv(file, sep="\t", names=columns)

        raw_df["label"] = raw_df.apply(
            lambda x: tweet_major_vote(x["annotation"]), axis=1)

        # # select and rename column
        # raw_df = raw_df.rename(columns={sent1_key: "sent1", sent2_key: "sent2"})
        raw_df = raw_df[["sent1", "sent2", "label"]]
        logger.info(f"Saving to {out_path}")
        raw_df.to_json(path_or_buf=out_path, orient='records', lines=True)


@add_start_docstrings("For caption type corpus,\
    to export two types of processed files:\
    1) sentence clusters\
    2) todo: sentenc pair with all positive label")
def process_msrvtt():
    task = "msrvtt"
    split = "train"
    restore_train = "/Users/timhuang/Desktop/paraphrase_datasets/Processed/msrvtt/train.jsonl"
    raw_file = DATASET_MAP[task][split]
    with open(raw_file) as f_in:
        all_annotations = json.load(f_in)["annotations"]
    caption_cluster = defaultdict(list)
    for caption in all_annotations:
        image_id = caption["image_id"]
        caption_cluster[image_id].append(caption["caption"])
    with open(restore_train, "w") as f_out:
        for image_id, caption_texts in caption_cluster.items():
            caption_texts = list(set(caption_texts))
            json.dump(
                {"image_id": image_id, "captions": caption_texts},
                f_out
            )
            f_out.write("\n")


def process_coco():
    task = "mscoco"
    split = "train"
    restore_train = "/Users/timhuang/Desktop/paraphrase_datasets/Processed/coco/train.jsonl"
    raw_file = DATASET_MAP[task][split]
    caption_cluster = defaultdict(list)
    with open(raw_file) as f_in:
        for json_s in f_in.readlines():
            content = json.loads(json_s)
            image_id = content["guid"]
            texts = content["texts"]
            for text in texts:
                caption_cluster[image_id].append(text)

    with open(restore_train, "w") as f_out:
        for image_id, caption_texts in caption_cluster.items():
            caption_texts = list(set(caption_texts))
            json.dump(
                {"image_id": image_id, "captions": caption_texts},
                f_out
            )
            f_out.write("\n")


def process_flickr():
    task = "flickr"
    split = "train"
    restore_train = "/Users/timhuang/Desktop/paraphrase_datasets/Processed/flickr/train.jsonl"
    raw_file = DATASET_MAP[task][split]
    caption_cluster = defaultdict(list)
    with open(raw_file) as f_in:
        for json_s in f_in.readlines():

            content = json.loads(json_s)
            image_id = content["guid"]
            texts = content["texts"]
            for text in texts:
                caption_cluster[image_id].append(text)

    with open(restore_train, "w") as f_out:
        for image_id, caption_texts in caption_cluster.items():
            caption_texts = list(set(caption_texts))
            json.dump(
                {"image_id": image_id, "captions": caption_texts},
                f_out
            )
            f_out.write("\n")


def process_split(mode="pos_only"):
    processed_file_dir = "/Users/timhuang/Desktop/paraphrase_datasets/Processed"
    if mode not in ["pos_only", "neg_only"]:
        raise NotImplementedError("Only pos_only mode is supported!")
    split_names = ["mnli_quora_split1.jsonl", "mnli_quora_split2.jsonl", "mnli_quora_split3.jsonl", "mnli_quora_split4.jsonl", "mnli_quora_split5.jsonl"]
    for filename in split_names:
        split_name = filename.split(".")[0]
        pos_only_filename = split_name + "_pos.jsonl" if mode == "pos_only"\
            else split_name + "_neg.jsonl"
        with open(os.path.join(processed_file_dir, filename)) as f_in,\
                open(os.path.join(processed_file_dir, pos_only_filename), "w") as f_out:
            for json_s in f_in.readlines():
                if len(json_s) == 0:
                    print(f"Empty line in: {filename}")
                    continue
                content = json.loads(json_s.strip())
                label = content["label"]
                if (mode == "pos_only" and label == 0)\
                        or (mode == "neg_only" and label == 1):
                    continue
                json.dump(content, f_out)
                f_out.write("\n")


if __name__ == "__main__":
    # process_mnli()
    # process_nli()
    # process_quora()
    # remap_mnli_test()
    # remap_stsb()
    # process_msrvtt()
    # process_flickr()
    # process_coco()
    # process_twitter()
    process_split(mode="pos_only")
