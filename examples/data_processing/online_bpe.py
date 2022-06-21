import argparse
import json
import sys
import time
import traceback
import requests
import os
from tqdm.auto import tqdm


vocab_size_map = {
    "english": 55000,
    "chinese": 70000,
    "spanish": 50000,
}


def batch_request_bpe_from_transmart(text_list, lang, url="https://dev.transmart.qq.com/api/nlu/tokenText"):
    headers = {}
    payload = {
        "header": {
            "api_key": "POJH1OKIDCckq9rrkkSQ",
            "user_name": "wx_tl2kvrt032qc"
        },
        "value": {
            "language": lang,
            "bpe_enable": True,
            "bpe_option": {
                "vocab_size": vocab_size_map[lang],
                "unk_if_exceed": False,
                "case_sensitive": True
            },
            "text_list": text_list

        }
    }

    try:
        res = requests.post(url, headers=headers,
                            data=json.dumps(payload).encode('utf-8'))
        res.encoding = 'utf-8'
        res_json = res.json()
        bpe_list = res_json['value']["text_list"]
    except KeyError:
        sys.stderr.write(traceback.format_exc())
        sys.stderr.write('####################')
        sys.stderr.write(text_list + '\tException\n')
    return bpe_list


def query_from_online(path_in, path_out, lang, batch_size=10):
    idx = 0
    batch = list()
    with open(path_in, 'r', encoding='utf-8') as fin, open(path_out, 'w', encoding='utf-8') as fout:
        all_ln = fin.readlines()
        progress_bar = tqdm(range(len(all_ln)))
        for line in all_ln:
            line_text = json.loads(line).get(args.sent_key)
            batch.append(line_text.strip('\n'))
            idx += 1
            if idx % batch_size == 0:
                bpe_result = batch_request_bpe_from_transmart(batch, lang)
                for res in bpe_result:
                    print(res, file=fout)               
                batch = list()
            progress_bar.update(1)
        if len(batch) > 0:
            bpe_result = batch_request_bpe_from_transmart(batch, lang)
            for res in bpe_result:
                print(res, file=fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='input file directory')
    parser.add_argument('--input', '-i', type=str, required=True, help='input file directory')
    parser.add_argument("--sent_key", type=str, default="text", help="The key to fetch src raw text from jsonl file")
    parser.add_argument('--out_dir', type=str, required=True, help='input file directory')
    parser.add_argument('--output', '-o', type=str, required=True, help='output file directory')
    parser.add_argument('--lang', '-l', type=str, required=True,
                        help='language, should be full name: english, chinese, spanish, etc.')
    args = parser.parse_args()
    print(args)
    path_in = os.path.join(args.data_dir, args.input)
    path_out = os.path.join(args.data_dir, args.output)
    lang = args.lang
    query_from_online(path_in, path_out, lang)
