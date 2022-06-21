import argparse
import sys
import os
import logging
import json
import pandas as pd
from typing import Dict, Callable
from collections import defaultdict


def spliting(args):
    in_file, pos_out, neg_out = os.path.join(args.data_dir, args.pred_filename),\
        os.path.join(args.data_dir, args.out_filename_pos), os.path.join(args.data_dir, args.out_filename_neg)
    with open(in_file) as f_in, open(pos_out, "w") as f_out1,\
            open(neg_out, "w") as f_out2:
        for json_s in f_in.readlines():
            content = json.loads(json_s)
            if str(content[args.sent_key]) == "1":
                json.dump(content, f_out1)
                f_out1.write("\n")
            else:
                json.dump(content, f_out2)
                f_out2.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="split into pos and neg")
    parser.add_argument("--data_dir", type=str, required=True, help="Model pred file dir")
    parser.add_argument("--pred_filename", type=str, required=True, help="Model pred filename")
    parser.add_argument("--sent_key", type=str, default="pred", help="Key name of preds in the original jsonl file")
    parser.add_argument("--out_filename_pos", type=str, default=None, help="Filename to save")
    parser.add_argument("--out_filename_neg", type=str, default=None, help="Filename to save")
    args = parser.parse_args()
    if ".jsonl" not in args.pred_filename:
        name_prefix = args.pred_filename
        args.pred_filename += ".jsonl"
    else:
        name_prefix, _ = args.pred_filename.split(".")
    args.out_filename_pos = args.out_filename_pos if args.out_filename_pos is not None else f"{name_prefix}-pos.jsonl"
    args.out_filename_neg = args.out_filename_neg if args.out_filename_neg is not None else f"{name_prefix}-neg.jsonl"
    # key_mappings = {args.pred_key: "label"}
    spliting(args)
