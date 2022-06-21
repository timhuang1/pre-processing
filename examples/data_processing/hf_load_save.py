import datasets
import os
import subprocess
import argparse
import transformers
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


def datasets_load_and_save(args):
    task_names = args.task_names
    task_names = task_names.split("__")
    cache_dir = "/apdcephfs/share_916081/timxthuang/cache"
    for task in task_names:
        sub_version = None
        if ":" in task:
            task, sub_version = task.split(":")
        re_save_dir = f"/apdcephfs/share_916081/timxthuang/bt_files/mono_en/{task}"
        os.makedirs(re_save_dir, exist_ok=True)
        # os.makedirs(re_save_dir, exist_ok=True)
        _dataset = load_dataset(task, cache_dir=cache_dir)
        for split in _dataset.keys():
            _dataset[split].to_json(f"/apdcephfs/share_916081/timxthuang/bt_files/mono_en/{task}/{task}_{split}.jsonl", lines=True)


def model_load_and_save(args):
    assert args.model_names, "args.model_names MUST be given for model_load_and_save"
    model_names = args.model_names.split("::")
    for hf_model in model_names:
        model_sub_dir = hf_model.replace("/", "_")
        model_dir = os.path.join(args.save_dir, model_sub_dir)
        os.makedirs(model_dir, exist_ok=True)
        model = AutoModel.from_pretrained(hf_model, cache_dir=args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(hf_model, cache_dir=args.cache_dir)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"{hf_model} download and save finished")


func_mapping = {
    "datasets_load_and_save": datasets_load_and_save,
    "model_load_and_save": model_load_and_save,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download datasets from huggingface and save to jsonl files")
    parser.add_argument("--func_name", type=str, required=True, help="The func to perform")
    parser.add_argument("--save_dir", type=str, default=True, help="The dir (usually on ceph) to save model/dataset")
    parser.add_argument("--cache_dir", type=str, default=True, help="The dir (usually on ceph) to save tmp downloaded model/dataset")
    parser.add_argument("--task_names", type=str, default=None, help="The datasets to load and save")
    parser.add_argument("--model_names", type=str, default=None, help="The hf pretrained models to load and save")
    
    args = parser.parse_args()
    func_mapping[args.func_name](args)
