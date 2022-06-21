import argparse
import logging
import json
import os
from copy import copy

logger = logging.getLogger(__name__)


def handle_one2many_preds(args):
    in_file = args.filename
    if "." not in in_file:
        in_file = f"{in_file}.jsonl"
    if args.out_filename is not None and "." not in args.out_filename:
        args.out_filename = f"{args.out_filename}.jsonl"
    file_prefix, extension = in_file.split(".")
    with open(os.path.join(args.data_dir, in_file), encoding="utf-8") as f_in:
        all_ln = f_in.readlines()
    all_examples = [json.loads(ln) for ln in all_ln]
    out_filename = args.out_filename if args.out_filename is not None\
        else f"{file_prefix}-filtered.{extension}"

    with open(os.path.join(args.data_dir, out_filename), "w", encoding="utf-8") as f_out:
        for example in all_examples:
            if len(example[args.check_key]) == 0:
                logger.info(f"empty check_key {example}")
                continue
            else:
                all_preds = list(set(example[args.check_key]))
                for pred in all_preds:
                    if len(pred) == 0:
                        continue
                    tmp_example = copy(example)
                    tmp_example[args.check_key] = pred
                    json.dump(
                        tmp_example,
                        f_out,
                        ensure_ascii=False,
                    )
                    f_out.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Removing empty pred lines")
    parser.add_argument("--data_dir", type=str, required=True, help="Target processing file dir")
    parser.add_argument("--filename", type=str, required=True, help="Target processing file")
    parser.add_argument("--check_key", type=str, default="pred", help="Target processing file")
    parser.add_argument("--out_filename", type=str, default=None, help="Target processing file")
    args = parser.parse_args()
    handle_one2many_preds(args)
