from pprint import pprint
import os


DATA_DIR = "/Users/timhuang/Desktop/paraphrase_datasets/"
PROCESSED_DIR = "/Users/timhuang/Desktop/paraphrase_datasets/Processed"

DATA_SET_SUBDIR = {
    "mnli": "NLI/multinli_1.0",
    "snli": "NLI/snli_1.0",
    "msrp": "MSRP",
    "quora": "Questions/quora-question-pairs",
    "stsb": "stsb",
    "mscoco": "captions",
    "msrvtt": "captions",
    "flickr": "captions",
    "tweets": "Tweets/paraphrase_dataset_emnlp2017",
}

ORI_DATAFILE = {
    "mnli": {
        "train": "multinli_1.0_train.txt",
        "val": "multinli_1.0_dev_matched.txt",
        # "test": "multinli_1.0_test_matched.txt",
        "val_mismatch": "multinli_1.0_dev_mismatched.txt",
        # "test_mismatch": "multinli_1.0_test_mismatched.txt",
    },
    "snli": {
        "train": "snli_1.0_train.txt",
        "val": "snli_1.0_dev.txt",
        "test": "snli_1.0_test.txt",
    },
    "msrp": {
        "train": "msr_paraphrase_train.txt",
        "test": "msr_paraphrase_test.txt",
    },
    "quora": {
        "train": "train.csv",
        "test": "test.csv",
    },
    "stsb": {
        "train": "train.jsonl",
        "val": "validation.jsonl",
        "test": "test.jsonl",
    },
    "mscoco": {
        "train": "coco_captions.jsonl",
    },
    "msrvtt": {
        "train": "MSR_VTT.json",
    },
    "flickr": {
        "train": "flickr30k_captions.jsonl",
    },
    "tweets": {
        "train": "Twitter_URL_Corpus_train.txt",
        "test": "Twitter_URL_Corpus_test.txt",
    }
}


PROCESSED_FILE = {
    "mnli": {
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
        "val_mismatch": "val_mismatch.jsonl",
        "test_mismatch": "test_mismatch.jsonl",
        "merge": "merge.jsonl",
    },
    "snli": {
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
        "merge": "merge.jsonl",
    },
    "quora": {
        "train": "train.jsonl",
        "test": "test.jsonl",
        "merge": "merge.jsonl",
    },
    "stsb": {
        "train": "train.jsonl",
        "val": "validation.jsonl",
        "test": "test.jsonl",
        "merge": "merge.jsonl",
    },
    "tweets": {
        "train": "train.jsonl",
        "test": "test.jsonl",
        "merge": "merge.jsonl",
    },
}


def merge_path():
    dataset_map = dict()
    for task, sub_set_name in ORI_DATAFILE.items():
        dataset_map[task] = dict()
        task_subdir = DATA_SET_SUBDIR[task]
        for split, file_name in sub_set_name.items():
            dataset_map[task][split] = os.path.join(
                DATA_DIR,
                task_subdir,
                file_name
            )
    return dataset_map


def merge_process_path():
    dataset_map = dict()
    for task, sub_set_name in PROCESSED_FILE.items():
        dataset_map[task] = dict()
        task_subdir = task
        os.makedirs(os.path.join(PROCESSED_DIR, task_subdir), exist_ok=True)
        for split, file_name in sub_set_name.items():
            dataset_map[task][split] = os.path.join(
                PROCESSED_DIR,
                task_subdir,
                file_name
            )
    return dataset_map


DATASET_MAP = merge_path()
PROCESSED_DATASET_MAP = merge_process_path()


if __name__ == "__main__":
    pprint(DATASET_MAP)
