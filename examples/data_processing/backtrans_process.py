import spacy
import argparse
import os
import linecache
import json
import copy
import subprocess
import random
import nltk
import spacy_fastlang
import benepar
import socket
import re
import torch
import torch.distributed as torch_dist
from collections import defaultdict, Counter
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from ftfy import fix_text
from pathlib import Path
from multiprocessing import Pool
from apted import APTED
from apted.helpers import Tree

CACHE_DIR = "/data1/cache"
# tokenizer = AutoTokenizer.from_pretrained("roberta-large")

STOP_WORDS = {'a', 'of', 'along', 'an', 'and', 'at', 'are', 'as', 'at', 'were', 'was', 'the', 'to', 'or', 'on', 'off', 'it', 'that', 'had'}


def mono_sent_split_file(pass_arg):
    filename, args = pass_arg
    print(f"{filename}, {args.major_process}")
    spacy_pipe = spacy.load("en_core_web_trf")
    filename_prefix = filename.split(".")[0]
    out_filename = f"{filename_prefix}-sents.jsonl"
    src_file, out_file = os.path.join(pass_arg.data_dir, filename),\
        os.path.join(pass_arg.out_dir, out_filename)
    ln_count = sum(1 for _ in Path(src_file).open(encoding="utf-8", errors='ignore').readlines())
    lang_code = "en"
    progress_bar = tqdm(range(ln_count), disable=not args.major_process)
    
    def spliting(line_text):
        line_text = fix_text(line_text)
        doc = spacy_pipe(line_text)
        valid_sent = list()
        for sent in doc.sents:
            reject = True
            if sent[-1].is_punct:
                has_noun = 2
                has_verb = 1
                for token in sent:
                    if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                        has_noun -= 1
                    elif token.pos_ == "VERB":
                        has_verb -= 1
                if has_noun < 1 and has_verb < 1:
                    reject = False
                    valid_sent.append(sent.text.strip())
            # if reject:
            #     print(sent.text.strip())
        return valid_sent

    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx in range(ln_count):
            # linecache starts at 1
            idx += 1
            line = linecache.getline(str(src_file), idx).rstrip("\n")
            content = json.loads(line)
            ln_text = content.get("text")
            ln_sents = spliting(ln_text)
            progress_bar.update(1)
            if not ln_sents:
                continue
            else:
                for sent in ln_sents:
                    example = dict(
                        lang_code=lang_code,
                        text=sent,
                    )
                    json.dump(
                        example,
                        f_out
                    )
                    f_out.write("\n")    
    return out_filename


def mono_sent_split_text_file(args):
    filename = args.sub_filename
    print(f"{filename}, {args.major_process}")
    spacy_pipe = spacy.load("en_core_web_trf")
    filename_prefix = filename.split(".")[0]
    out_filename = f"{filename_prefix}-sents.jsonl"
    src_file, out_file = os.path.join(args.data_dir, filename),\
        os.path.join(args.data_dir, out_filename)
    ln_count = sum(1 for _ in Path(src_file).open(encoding="utf-8", errors='ignore').readlines())
    # lang_code = "en"
    progress_bar = tqdm(range(ln_count), disable=not args.major_process)
    
    def spliting(line_text):
        line_text = fix_text(line_text)
        doc = spacy_pipe(line_text)
        valid_sent = list()
        for sent in doc.sents:
            reject = True
            if sent[0].is_title and sent[-1].is_punct:
                valid_len = sum([token.pos_ != "PUNCT" and token.pos_ != "NUM" for token in sent])
                if valid_len < 8:
                    continue
                has_noun = 2
                has_verb = 1
                for token in sent:
                    if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                        has_noun -= 1
                    elif token.pos_ == "VERB":
                        has_verb -= 1
                if has_noun < 1 and has_verb < 1:
                    reject = False
                    valid_sent.append(fix_text(sent.text.strip()))
            # if reject:
            #     print(sent.text.strip())
        return valid_sent

    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx in range(ln_count):
            # linecache starts at 1
            idx += 1
            ln_text = linecache.getline(str(src_file), idx).rstrip("\n")
            # content = json.loads(line)
            # ln_text = content.get("text")
            ln_sents = spliting(ln_text)
            progress_bar.update(1)
            if not ln_sents:
                continue
            else:
                for sent in ln_sents:
                    example = dict(
                        # lang_code=lang_code,
                        text=sent,
                    )
                    json.dump(
                        example,
                        f_out
                    )
                    # f_out.write(fix_text(sent))
                    f_out.write("\n")    
    return out_filename


def compute_tree_edit(args):
    filename = args.sub_filename
    out_filename = f"{filename}.tmp"
    src_file, out_file = os.path.join(args.data_dir, filename),\
        os.path.join(args.data_dir, out_filename)
    assert len(args.text_cols.split("::")) == 2, f"Invalid text_cols: {args.text_cols}"
    ref_col, pred_col = args.text_cols.split("::")
    metric_name = "tree_edit"

    def normalize_tree(tree_string, max_depth=3):
        res = []
        depth = -1
        leaf = False
        for c in tree_string:
            if c in ['{', '}']:
                continue
            if c == '(':
                leaf = False
                depth += 1

            elif c == ')':
                leaf = False
                depth -= 1
                if depth < max_depth:
                    res.append('}')
                    continue
                    
            elif c == ' ':
                leaf = True
                continue

            if depth <= max_depth and not leaf and c != ')':
                res.append(c if c != '(' else '{')
            
        return ''.join(res)    

    def tree_edit_distance(lintree1, lintree2):
        tree1 = Tree.from_text(lintree1)
        tree2 = Tree.from_text(lintree2)
        n_nodes_t1 = lintree1.count('{')
        n_nodes_t2 = lintree2.count('{')
        apted = APTED(tree1, tree2)
        ted = apted.compute_edit_distance()
        return ted / (n_nodes_t1 + n_nodes_t2)

    def dist(pair):
        p_tree_n = normalize_tree(pair[0], max_depth=3)
        r_tree_n = normalize_tree(pair[1], max_depth=3)
        ted = tree_edit_distance(p_tree_n, r_tree_n)
        return ted

    def get_div(sent1, sent2, mode="bow_edit"):
        sent1, sent2 =\
            re.sub('[^A-Za-z0-9]+', ' ', sent1),\
            re.sub('[^A-Za-z0-9]+', ' ', sent2)
        sent1 = sent1.strip().lower().split(" ")
        sent2 = sent2.strip().lower().split(" ")
        if mode == "bow_edit":
            sent1 = [w.lower() if w.lower() not in STOP_WORDS else "" for w in sent1]
            sent2 = [w.lower() if w.lower() not in STOP_WORDS else "" for w in sent2]
            sent1 = list(filter(len, sent1))
            sent2 = list(filter(len, sent2))
            sent1 = " ".join(sorted(sent1))
            sent2 = " ".join(sorted(sent2))
            div_score = nltk.edit_distance(sent1, sent2) / max(len(sent1), len(sent2))
            return div_score

    with open(src_file, encoding="utf-8") as f_in:
        jsonl_lns = f_in.readlines()

    progress_bar = tqdm(range(len(jsonl_lns)), disable=not args.major_process)
    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx in range(len(jsonl_lns)):
            progress_bar.update(1)
            content = json.loads(jsonl_lns[idx])
            # content["word_edit"] = word_edit_scores[idx]
            ref_text, pred_text = content["src_en"], content["pred_en"]
            # # content[metric_name] = scores[idx]
            # content["ref_tree"] = refs[idx]
            # content["pred_tree"] = preds[idx]
            bow_edit = get_div(ref_text, pred_text)
            ref_tree = content.pop(ref_col)
            pred_tree = content.pop(pred_col)
            content[metric_name] = dist((ref_tree, pred_tree))
            content["bow_edit"] = bow_edit
            json.dump(
                content,
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")
    return out_filename


def filter_editable(args):
    keep_col = args.keep_col.split("::") if args.keep_col else None
    src_editable_col, tgt_editable_col, edit_mapping_col = args.text_cols.split("::")
    filename = args.sub_filename
    out_filename = f"{filename}.tmp"
    src_file, out_file = os.path.join(args.data_dir, filename),\
        os.path.join(args.data_dir, out_filename)
    with open(src_file) as f_in1:
        jsonl_lns = f_in1.readlines()
    ln_count = len(jsonl_lns)

    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    nlp = spacy.load("en_core_web_sm")
    STOP_WORDS = nlp.Defaults.stop_words

    with open(out_file, "w", encoding="utf-8") as f_out:
        progress_bar = tqdm(range(ln_count), disable=not args.major_process)
        for idx in range(ln_count):
            progress_bar.update(1)
            json_content = json.loads(jsonl_lns[idx]) 
            src_editable, tgt_editable, edit_mapping =\
                json_content[src_editable_col],\
                json_content[tgt_editable_col],\
                json_content[edit_mapping_col]
            valid_edit_mapping = list()
            for map_pair in edit_mapping:
                src_word, tgt_word = map_pair
                # add stop-word filtering
                if src_word.lower() in STOP_WORDS or tgt_word.lower() in STOP_WORDS:
                    src_editable.remove(src_word) if src_word in src_editable else None
                    tgt_editable.remove(tgt_word) if tgt_word in tgt_editable else None
                    continue
                src_stem, tgt_stem = stemmer.stem(src_word.lower()), stemmer.stem(tgt_word.lower())
                if src_stem == tgt_stem:
                    src_editable.remove(src_word) if src_word in src_editable else None
                    tgt_editable.remove(tgt_word) if tgt_word in tgt_editable else None
                    continue
                valid_edit_mapping.append((src_word, tgt_word))
            src_editable = [w for w in src_editable if w not in STOP_WORDS]
            tgt_editable = [w for w in tgt_editable if w not in STOP_WORDS]
            if keep_col is not None:
                _save_dict = {key: json_content[key] for key in keep_col}
            else:
                _save_dict = json_content
            _save_dict[src_editable_col] = copy.copy(src_editable)
            _save_dict[tgt_editable_col] = copy.copy(tgt_editable)
            _save_dict[edit_mapping_col] = copy.copy(valid_edit_mapping)
            json.dump(
                _save_dict,
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")
    return out_filename


def extract_editable(args):
    filename = args.sub_filename
    out_filename = f"{filename}.tmp"
    # keep_col = args.keep_col.split("::") if args.keep_col else None
    src_col, tgt_col = args.text_cols.split("::")
    src_file, out_file = os.path.join(args.data_dir, filename),\
        os.path.join(args.data_dir, out_filename)

    VALID_POS_TYPES = ['PRON', 'NOUN', 'ADJ', 'VERB', 'ADV']
    nlp = spacy.load("en_core_web_sm")
    
    with open(src_file, encoding="utf-8") as f_in:
        jsonl_lns = f_in.readlines()
    ln_count = len(jsonl_lns)
    progress_bar = tqdm(range(ln_count), disable=not args.major_process)
    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx in range(ln_count):
            progress_bar.update(1)
            json_content = json.loads(jsonl_lns[idx])
            src_editable, tgt_editable, edit_mapping = list(), list(), list()
            # get src and tgt token pos type
            src_text, tgt_text = json_content[src_col], json_content[tgt_col]
            src_text, tgt_text = re.sub('[^A-Za-z0-9]+', ' ', src_text), re.sub('[^A-Za-z0-9]+', ' ', tgt_text)
            src_res = [(token.text, token.pos_) for token in nlp(src_text)]
            tgt_res = [(token.text, token.pos_) for token in nlp(tgt_text)]
            
            # iterate over idx diff and add token to src/tgt set respectively
            src_idx_mapping = json_content["src_idx_mapping"]
            src_idx_mapping = [(pair.split("-")[0], pair.split("-")[1]) for pair in src_idx_mapping.split()]
            # all_src_edited, all_tgt_edited = list(), list()
            for diff in json_content["idx_diff"]:
                src_idx, tgt_idx = src_idx_mapping[int(diff)]
                src_valid, tgt_valid = False, False
                if src_idx != "None":
                    _src_token_text, _src_token_pos = src_res[int(src_idx)]
                    # all_src_edited.append(_src_token_text)
                    if _src_token_pos in VALID_POS_TYPES:
                        # src_editable.add(_src_token_text)
                        src_editable.append((_src_token_text, int(src_idx)))
                        src_valid = True
                if tgt_idx != "None":
                    _tgt_token_text, _tgt_token_pos = tgt_res[int(tgt_idx)]
                    # all_tgt_edited.append(_tgt_token_text)
                    if _tgt_token_pos in VALID_POS_TYPES:
                        # tgt_editable.add(_tgt_token_text)
                        tgt_editable.append((_tgt_token_text, int(tgt_idx)))
                        tgt_valid = True
                if src_valid and tgt_valid:
                    edit_mapping.append((_src_token_text, _tgt_token_text))

            src_editable.sort(key=lambda x: x[1])
            tgt_editable.sort(key=lambda x: x[1])
            src_editable = [pair[0] for pair in src_editable]
            tgt_editable = [pair[0] for pair in tgt_editable]
            src_editable = list(dict.fromkeys(src_editable))
            tgt_editable = list(dict.fromkeys(tgt_editable))
            src_editable = [word for word in src_editable if word not in tgt_text.split()]
            tgt_editable = [word for word in tgt_editable if word not in src_text.split()]
            # if keep_col is not None:
            #     _save_dict = {key: json_content[key] for key in keep_col}
            # else:
            #     _save_dict = json_content
            _save_dict = json_content
            _save_dict["src_editable"] = src_editable
            _save_dict["tgt_editable"] = tgt_editable
            _save_dict["edit_mapping"] = edit_mapping
            json.dump(
                _save_dict,
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")
    return out_filename


def pred_res_select(args):
    filename = args.sub_filename
    out_filename = f"{filename}.tmp"
    src_file, out_file = os.path.join(args.data_dir, filename),\
        os.path.join(args.data_dir, out_filename)
    div_mode = "bow_edit"

    def get_div(sent1, sent2, mode="bow_edit"):
        sent1, sent2 =\
            re.sub('[^A-Za-z0-9]+', ' ', sent1),\
            re.sub('[^A-Za-z0-9]+', ' ', sent2)
        sent1 = sent1.strip().lower().split(" ")
        sent2 = sent2.strip().lower().split(" ")
        if mode == "word_edit":
            div_score = nltk.edit_distance(sent1, sent2) / max(len(sent1), len(sent2))
            return div_score
        elif mode == "bow_edit":
            sent1 = [w.lower() if w.lower() not in STOP_WORDS else "" for w in sent1]
            sent2 = [w.lower() if w.lower() not in STOP_WORDS else "" for w in sent2]
            sent1 = list(filter(len, sent1))
            sent2 = list(filter(len, sent2))
            sent1 = " ".join(sorted(sent1))
            sent2 = " ".join(sorted(sent2))
            div_score = nltk.edit_distance(sent1, sent2) / max(len(sent1), len(sent2))
            return div_score
        else:
            raise NotImplementedError(f"{mode} is not a valid div mode, options are ['word_edit', 'bow_edit']")

    with open(src_file, encoding="utf-8") as f_in:
        jsonl_lns = f_in.readlines()
    progress_bar = tqdm(range(len(jsonl_lns)), disable=not args.major_process)
    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx in range(len(jsonl_lns)):
            progress_bar.update(1)
            content = json.loads(jsonl_lns[idx])
            ref_text = content["src_en"]
            if args.no_sort:
                all_candidates = content[args.text_cols]
                all_candidates.sort(key=lambda x: get_div(x, ref_text, mode=div_mode), reverse=True)
                content["selected_pred_bow"] = all_candidates[0]
            else:
                eval_text = content[args.text_cols]
                content["bow_edit"] = get_div(eval_text, ref_text, mode=div_mode)
            json.dump(content, f_out, ensure_ascii=False)
            f_out.write("\n")
    
    return out_filename


def diversity_measure(args):
    # import torch.distributed as torch_dist
    torch_dist.init_process_group(
        "nccl",
        init_method=args.url,
        rank=args.local_rank,
        world_size=args.nsplit
    )
    torch.cuda.set_device(args.local_rank)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    filename = args.sub_filename
    # filename_prefix = filename.split(".")[0]
    # out_filename = f"{filename_prefix}-measure.tmp.jsonl"
    out_filename = args.out_sub_filename
    src_file, out_file = os.path.join(args.data_dir, filename),\
        os.path.join(args.data_dir, out_filename)
    assert len(args.text_cols.split("::")) == 2, f"Invalid text_cols: {args.text_cols}"
    ref_col, eval_col = args.text_cols.split("::")
    metric_name = "tree_edit"
    # ln_count = sum(1 for _ in Path(src_file).open(encoding="utf-8", errors='ignore').readlines())
    # progress_bar = tqdm(range(ln_count), disable=not args.major_process)

    def normalize_tree(tree_string, max_depth=3):
        res = []
        depth = -1
        leaf = False
        for c in tree_string:
            if c in ['{', '}']:
                continue
            if c == '(':
                leaf = False
                depth += 1

            elif c == ')':
                leaf = False
                depth -= 1
                if depth < max_depth:
                    res.append('}')
                    continue
                    
            elif c == ' ':
                leaf = True
                continue

            if depth <= max_depth and not leaf and c != ')':
                res.append(c if c != '(' else '{')
            
        return ''.join(res)

    def tree_edit_distance(lintree1, lintree2):
        tree1 = Tree.from_text(lintree1)
        tree2 = Tree.from_text(lintree2)
        n_nodes_t1 = lintree1.count('{')
        n_nodes_t2 = lintree2.count('{')
        apted = APTED(tree1, tree2)
        ted = apted.compute_edit_distance()
        return ted / (n_nodes_t1 + n_nodes_t2)

    def get_tree_string(doc):
        return next(iter(doc.sents))._.parse_string

    def dist(pair):
        p_tree_n = normalize_tree(pair[0], max_depth=3)
        r_tree_n = normalize_tree(pair[1], max_depth=3)
        ted = tree_edit_distance(p_tree_n, r_tree_n)
        return ted

    def get_word_edit(sent1, sent2):
        sent1 = re.sub('[^A-Za-z0-9]+', ' ', sent1)
        sent2 = re.sub('[^A-Za-z0-9]+', ' ', sent2)
        sent1 = sent1.lower().split()
        sent2 = sent2.lower().split()
        return nltk.edit_distance(sent1, sent2) / max(len(sent1), len(sent2))

    with open(src_file, encoding="utf-8") as f_in:
        jsonl_lns = f_in.readlines()

    # benepar.download('benepar_en3')
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_sm')
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    all_ref_text = [json.loads(ln)[ref_col] for ln in jsonl_lns]
    all_eval_text = [json.loads(ln)[eval_col] for ln in jsonl_lns]
    word_edit_scores = [get_word_edit(ref, pred) for ref, pred in zip(all_ref_text, all_eval_text)]
    with nlp.select_pipes(enable=["parser", "benepar"]):
        preds = list(tqdm(nlp.pipe(all_eval_text, batch_size=256), total=len(all_eval_text), desc="syntdiv:parse_preds", disable=(args.local_rank != 0)))
        preds = list(map(get_tree_string, preds))

        refs = list(tqdm(nlp.pipe(all_ref_text, batch_size=256), total=len(all_ref_text), desc="syntdiv:parse_refs", disable=(args.local_rank != 0)))
        refs = list(map(get_tree_string, refs))

    # scores = list(tqdm(map(dist, zip(preds, refs)), total=len(preds), desc="syntdiv:calc_dist"))
    
    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx in range(len(jsonl_lns)):
            content = json.loads(jsonl_lns[idx])
            content["word_edit"] = word_edit_scores[idx]
            
            # content[metric_name] = scores[idx]
            content["ref_tree"] = refs[idx]
            content["pred_tree"] = preds[idx]
            json.dump(
                content,
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")

    return out_filename


def backtrans_filter(args):
    filename = args.sub_filename
    print(f"{filename}, {args.major_process}")
    filename_prefix = filename.split(".")[0]
    high_edit_out, low_edit_out, trash_out =\
        f"{filename_prefix}-high_edit.jsonl",\
        f"{filename_prefix}-low_edit.jsonl",\
        f"{filename_prefix}-trash.jsonl"
    src_file, high_outfile, low_outfile, trash_outfile =\
        os.path.join(args.data_dir, filename),\
        os.path.join(args.data_dir, high_edit_out),\
        os.path.join(args.data_dir, low_edit_out),\
        os.path.join(args.data_dir, trash_out)

    src_col, tgt_col = args.text_cols.split("::")

    ln_count = sum(1 for _ in Path(src_file).open(encoding="utf-8", errors='ignore').readlines())
    progress_bar = tqdm(range(ln_count), disable=not args.major_process)
    spacy_pipe = spacy.load("en_core_web_sm")
    spacy_pipe.add_pipe("language_detector")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{CACHE_DIR}/roberta-large")
    # tokenizer = AutoTokenizer.from_pretrained(f"roberta-large")

    def get_div(sent1, sent2):
        div_score = nltk.edit_distance(sent1, sent2) / max(len(sent1), len(sent2))
        return div_score

    def not_valid(sent1, sent2, sent1_text, sent2_text):
        if len(sent1_text) > 500 or len(sent2_text) > 500:
            return True

        if max([len(w) for w in sent1]) > 25 or max([len(w) for w in sent2]) > 25:
            return True

        if len(sent1_text) - len(" ".join(sent1)) > 15 or\
                len(sent2_text) - len(" ".join(sent2)) > 15:
            return True

        if "\\u" in f"{sent1_text} {sent2_text}":
            return True

        if len(tokenizer(sent1_text)["input_ids"]) > 400 or\
                len(tokenizer(sent1_text)["input_ids"]) > 400:
            return True

        max_len, min_len = max(len(sent1), len(sent2)), min(len(sent1), len(sent2))
        if max_len > 125 or min_len < 5:
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

    with open(high_outfile, "w", encoding="utf-8") as fOut_high,\
            open(low_outfile, "w", encoding="utf-8") as fOut_low,\
            open(trash_outfile, "w", encoding="utf-8") as fOut_trash:
        for idx in range(ln_count):
            # linecache starts at 1
            idx += 1
            ln_text = linecache.getline(str(src_file), idx).rstrip("\n")
            # get sents
            sent1_text, sent2_text = json.loads(ln_text).get(src_col), json.loads(ln_text).get(tgt_col)
            progress_bar.update(1)
            if not sent1_text or not sent2_text:
                # or (len(sent1_text) > 420) or (len(sent2_text) > 420)
                fOut_trash.write(ln_text)
                fOut_trash.write("\n")
                continue

            sent1 = re.sub('[^A-Za-z0-9]+', ' ', sent1_text)
            sent2 = re.sub('[^A-Za-z0-9]+', ' ', sent2_text)

            sent1 = sent1.strip().lower().split(" ")
            sent2 = sent2.strip().lower().split(" ")
            if not_valid(sent1, sent2, sent1_text, sent2_text):
                fOut_trash.write(ln_text)
                fOut_trash.write("\n")
                continue

            # measure div and other heuristic rules (labeled as tag)
            #   tag: high-edit; low-edit; clean (length short/long or misalighn, `< unk >` issue 
            div_score = get_div(sent1, sent2)
            if div_score > 0.25:
                fOut_high.write(ln_text)
                fOut_high.write("\n")
            else:
                fOut_low.write(ln_text)
                fOut_low.write("\n")
    return high_edit_out, low_edit_out, trash_out


def file_merging(work_dir, filenames, out_filename):
    filenames = [str(os.path.join(work_dir, filename)) for filename in filenames]
    out_file = str(os.path.join(work_dir, out_filename))
    command = "cat " + " ".join(filenames) + " > " + out_file
    print(command)
    subprocess.call(command, shell=True)


def dual_thread_extract(args):
    assert args.filename1 is not None and args.filename2 is not None,\
        f"filename1 {args.filename1} filename2 {args.filename2} cannot be None"
    sub_args1 = copy.deepcopy(args)
    sub_args1.filename = args.filename1
    sub_args2 = copy.deepcopy(args)
    sub_args2.filename = args.filename2
    with Pool(2) as p:
        p.map(single_thread_extract, [sub_args1, sub_args2])


def fix_text_only(args):
    name_prefix, extension = args.sub_filename.split(".")
    out_filename = f"{name_prefix}-inter.{extension}"
    in_file, out_file = os.path.join(args.data_dir, args.sub_filename),\
        os.path.join(args.data_dir, out_filename)
    ln_count = sum(1 for _ in Path(in_file).open(encoding="utf-8", errors='ignore').readlines())
    
    spacy_pipe = spacy.load("en_core_web_sm")

    progress_bar = tqdm(range(ln_count), disable=not args.major_process)
    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx in range(ln_count):
            progress_bar.update(1)
            # linecache starts at 1
            idx += 1
            line = linecache.getline(str(in_file), idx).rstrip("\n")
            if extension == "jsonl":
                content = json.loads(line)
                ln_text = content.get("text")
            elif extension == "txt":
                ln_text = line.strip()
            clean_text = fix_text(ln_text)
            doc = spacy_pipe(clean_text)
            valid_len = sum([token.pos_ != "PUNCT" and token.pos_ != "NUM" for token in doc])
            if valid_len < 10:
                continue
            if "\\u" in ln_text:
                print(f"{ln_text} is still dirty")
                continue
            if extension == "jsonl":
                content["text"] = clean_text
                json.dump(
                    content,
                    f_out
                )
            elif extension == "txt":
                f_out.write(clean_text)
            f_out.write("\n")
    return out_filename


def run_split(args):
    sub_files = args.sub_files
    sub_files = sub_files.split(";")
    file_names = list(filter(len, sub_files))
    assert args.nsplit == len(file_names), f"file num {len(file_names)} and nsplit {args.nsplit} must be the same"
    pass_args = list()
    assert args.func_name in func_name_mapping, f"{args.func_name} is not defined or not added to mapping"
    run_func = func_name_mapping[args.func_name]
    for idx in range(args.nsplit):
        sub_args = copy.deepcopy(args)
        sub_args.major_process = (idx == 0)
        print(sub_args.major_process)
        sub_args.sub_filename = file_names[idx]
        pass_args.append(sub_args)
    with Pool(args.nsplit) as p:
        all_outfiles = p.map(run_func, pass_args)
    
    post_action_mapping[args.post_action](args, file_names, all_outfiles)


def run_gpu_split(args):
    sub_files = args.sub_files
    sub_files = sub_files.split(";")
    file_names = list(filter(len, sub_files))
    assert args.nsplit == len(file_names), f"file num {len(file_names)} and nsplit {args.nsplit} must be the same"
    out_sub_filenames = [f"{name}.tmp" for name in file_names]
    
    args.sub_filenames = copy.deepcopy(file_names)
    args.out_sub_filenames = copy.deepcopy(out_sub_filenames)

    # Pick a free port
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        url = "tcp://localhost:" + str(port)
        args.url = url
    ctx = torch.multiprocessing.spawn(
        process_fn,
        args=(args,),
        nprocs=args.nsplit,
        join=True,
    )

    # try:
    #     torch.distributed.barrier()
    # except Exception as e:
    #     pass
    
    post_action_mapping[args.post_action](args, file_names, out_sub_filenames)


# Wrap processing function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    local_args.sub_filename = args.sub_filenames[rank]
    local_args.out_sub_filename = args.out_sub_filenames[rank]
    run_func = func_name_mapping[args.func_name]
    run_func(local_args)
    # diversity_measure(local_args)


def merge_to_single(args, ori_part_files, all_outfiles):
    out_filename = args.out_filename if args.out_filename is not None else "post_cat_file.jsonl"
    file_merging(args.data_dir, all_outfiles, out_filename)
    inter_files = [str(os.path.join(args.data_dir, filename)) for filename in all_outfiles]
    part_files = [str(os.path.join(args.data_dir, filename)) for filename in ori_part_files]
    for inter_file in inter_files + part_files:
        command = "rm " + inter_file
        print(command)
        subprocess.call(command, shell=True)


def merge_to_multiple(args, ori_part_files, all_outfiles):
    num_per_split = len(all_outfiles[0])
    all_sub_split = list()
    for idx in range(num_per_split):
        all_sub_split.append([file[idx] for file in all_outfiles])

    out_filename = [f"post_cat_file{_idx}.jsonl" for _idx in range(num_per_split)]
    if args.out_filename is not None:
        given_out_filename = args.out_filename.split("::")
        if len(out_filename) != num_per_split:
            print(f"Warning: num of given out_filenames {args.out_filename} not match {num_per_split}")
        else:
            out_filename = given_out_filename
    for _outfiles, _outfilene in zip(all_sub_split, out_filename):
        file_merging(args.data_dir, _outfiles, _outfilene)
    inter_files = [str(os.path.join(args.data_dir, filename)) for sub_split in all_sub_split for filename in sub_split]
    part_files = [str(os.path.join(args.data_dir, filename)) for filename in ori_part_files]
    for inter_file in inter_files + part_files:
        command = "rm " + inter_file
        print(command)
        subprocess.call(command, shell=True)


def single_thread_extract(args):
    data_dir = "/apdcephfs/share_916081/timxthuang/bt_files/mono_en/msmarco"
    out_dir = "/apdcephfs/share_916081/timxthuang/bt_files/mono_en/msmarco_test"
    filename = args.filename
    filename_prefix = filename.split(".")[0]

    out_filename = f"{filename_prefix}-sents.jsonl"
    src_file = os.path.join(data_dir, filename) 
    out_file = os.path.join(out_dir, out_filename)

    spacy_pipe = spacy.load("en_core_web_trf")

    def spliting(pipe, line_text):
        line_text_fix = fix_text(line_text)
        doc = pipe(line_text_fix)
        valid_sent = list()
        for sent in doc.sents:
            reject = True
            if sent[-1].is_punct:
                has_noun = 2
                has_verb = 1
                for token in sent:
                    if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                        has_noun -= 1
                    elif token.pos_ == "VERB":
                        has_verb -= 1
                if has_noun < 1 and has_verb < 1:
                    reject = False
                    valid_sent.append(fix_text(sent.text.strip()))
            # if reject:
            #     print(sent.text.strip())
        return valid_sent

    ln_count = sum(1 for _ in Path(src_file).open(encoding="utf-8", errors='ignore').readlines())
    # ln_count = 2000
    all_sents = list()
    progress_bar = tqdm(range(ln_count))

    for idx in range(ln_count):
        # linecache starts at 1
        idx += 1
        line = linecache.getline(str(src_file), idx).rstrip("\n")
        content = json.loads(line)
        ln_text = content.get("text")
        ln_sents = spliting(spacy_pipe, ln_text)
        if not ln_sents:
            continue
        else:
            all_sents.append(ln_sents)
        progress_bar.update(1)

    progress_bar = tqdm(range(ln_count))
    with open(out_file, "w", encoding="utf-8") as f_out:
        for sents in all_sents:
            for sent in sents:
                example = dict(
                    # ori_para=ln_text,
                    # ln_sents=ln_sents,
                    text=sent,
                )
                json.dump(
                    example,
                    f_out
                )
                f_out.write("\n")
        progress_bar.update(1)


def split_inspect():
    spacy_pipe1 = spacy.load("en_core_web_sm")
    spacy_pipe2 = spacy.load("en_core_web_trf")
    data_dir = "/Users/timhuang/Desktop/pyprojects/beir/examples/dataset/msmarco"
    out_dir = "/Users/timhuang/Desktop/"
    src_file = os.path.join(data_dir, "corpus_1m.jsonl") 
    out_file = os.path.join(out_dir, "split_inspect.jsonl")

    def spliting(pipe, line_text):
        doc = pipe(line_text)
        valid_sent = list()
        for sent in doc.sents:
            reject = True
            if sent[-1].is_punct:
                has_noun = 2
                has_verb = 1
                for token in sent:
                    if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                        has_noun -= 1
                    elif token.pos_ == "VERB":
                        has_verb -= 1
                if has_noun < 1 and has_verb < 1:
                    reject = False
                    valid_sent.append(fix_text(sent.text.strip()))
            # if reject:
            #     print(sent.text.strip())
        return valid_sent

    ln_start, ln_end = 10000, 10100 
    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx in range(ln_start, ln_end + 1):
            line = linecache.getline(str(src_file), idx).rstrip("\n")
            content = json.loads(line)
            ln_text = content.get("text")
            ln_sents1 = spliting(spacy_pipe1, ln_text)
            ln_sents2 = spliting(spacy_pipe2, ln_text)
            example = dict(
                ori_para=ln_text,
                ln_sents1=ln_sents1,
                ln_sents2=ln_sents2,
            )
            json.dump(
                example,
                f_out
            )
            f_out.write("\n")


def single_task():
    parser = argparse.ArgumentParser(
        description="Extracting well-formed sentences from passages")
    # parser.add_argument("--sub_files", type=str, required=True, help="Names of the split files, seperated by ';'")
    # parser.add_argument("--nsplit", type=int, required=True, help="Num of processes to run simultaneously")
    parser.add_argument("--data_dir", type=str, required=True, help="Names of the split files dir")
    parser.add_argument("--filename", type=str, default=None, help="Target processing file")
    # parser.add_argument("--filename1", type=str, default=None, help="Target processing file1")
    # parser.add_argument("--filename2", type=str, default=None, help="Target processing file2")
    parser.add_argument("--func_name", type=str, required=True, help="Function name applied to each sample")
    args = parser.parse_args()
    func_name_mapping[args.func_name](args)

    # single_thread_extract(args)
    # dual_thread_extract(args)


def entity_mask(args):
    filename = args.sub_filename
    filename_prefix = filename.split(".")[0]
    sent1_key, sent2_key = "src_en", "pred_en"
    out_filename = f"{filename_prefix}-add_mask.jsonl"
    src_file, out_file = os.path.join(args.data_dir, filename),\
        os.path.join(args.data_dir, out_filename)
    ln_count = sum(1 for _ in Path(src_file).open(encoding="utf-8", errors='ignore').readlines())

    spacy_pipe = spacy.load("en_core_web_lg")
    # entity_store = defaultdict(list)

    def _replace_entity(text):
        ent_counter = defaultdict(int)
        mask_to_text = dict()
        doc = spacy_pipe(text)
        ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
        if not ents:
            return text, mask_to_text 
        else:
            for ent_info in ents[::-1]:
                ent_text, s_char, e_char, ent_label = ent_info
                if ent_label == "NORP":
                    ent_label = "GPE"
                # entity_store[ent_label].append(ent_text)
                text = text[:s_char] + f"{ent_label}-{ent_counter[ent_label]}" + text[e_char:]
                mask_to_text[f"{ent_label}-{ent_counter[ent_label]}"] = ent_text
                ent_counter[ent_label] += 1
            return text, mask_to_text

    progress_bar = tqdm(range(ln_count), disable=not args.major_process)
    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx in range(ln_count):
            idx += 1
            ln_text = linecache.getline(str(src_file), idx).rstrip("\n")
            content = json.loads(ln_text)
            sent1, sent2 = content.get(sent1_key), content.get(sent2_key)
            sent1, mask_to_text1 = _replace_entity(sent1)
            sent2, mask_to_text2 = _replace_entity(sent2)
            content[f"{sent1_key}_mask"] = sent1
            content[f"{sent1_key}_mask2text"] = mask_to_text1
            content[f"{sent2_key}_mask"] = sent2
            content[f"{sent2_key}_mask2text"] = mask_to_text2
            json.dump(
                content,
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")
            progress_bar.update(1)
        # json.dump(entity_store, f_out)
        # f_out.write("\n")

    return str(out_file)


def entity_mask_align(args):
    filename = args.sub_filename
    filename_prefix = filename.split(".")[0]
    sent1_key, sent2_key = "src_en", "pred_en"
    mask2t_key1, mask2t_key2 = f"{sent1_key}_mask2text", f"{sent2_key}_mask2text"
    # out_filename = f"{filename_prefix}-add_mask.jsonl.tmp"
    src_file = os.path.join(args.data_dir, filename)

    match_out, non_match_out =\
        f"{filename_prefix}-match.jsonl",\
        f"{filename_prefix}-non_match.jsonl",

    match_outfile, non_match_outfile =\
        os.path.join(args.data_dir, match_out),\
        os.path.join(args.data_dir, non_match_out),\

    cared_types = ["EVENT", "FAC", "GPE", "LAW", "LOC", "ORDINAL", "ORG", "PERSON", "WORK_OF_ART"]
    
    def _entity_align(mapping1, mapping2):
        map1_count = [_key.split("-")[0] for _key in mapping1.keys()]
        map2_count = [_key.split("-")[0] for _key in mapping2.keys()]
        c_1, c_2 = Counter(map1_count), Counter(map2_count)
        # loose version:
        c_1_all, c_2_all = 0, 0
        for e_type in cared_types:
            c_1_all += c_1[e_type]
            c_2_all += c_2[e_type]
        if c_1_all != c_2_all:
            return False

        # # strict version
        # for key in c_1:
            # if key not in cared_types:
            #     continue
            # if key not in c_2 or c_1[key] != c_2[key]:
            #     return False
        return True

    ln_count = sum(1 for _ in Path(src_file).open(encoding="utf-8", errors='ignore').readlines())
    progress_bar = tqdm(range(ln_count))
    
    with open(match_outfile, "w", encoding="utf-8") as fOut_match,\
            open(non_match_outfile, "w", encoding="utf-8") as fOut_non_match:
        for idx in range(ln_count):
            progress_bar.update(1)
            idx += 1
            ln_text = linecache.getline(str(src_file), idx).rstrip("\n")
            content = json.loads(ln_text)
            sent1_mask2text, sent2_mask2text =\
                content[mask2t_key1], content[mask2t_key2]
            if not _entity_align(sent1_mask2text, sent2_mask2text):
                fOut_non_match.write(ln_text)
                fOut_non_match.write("\n")
                continue
            else:
                fOut_match.write(ln_text)
                fOut_match.write("\n")
    return match_out, non_match_out


def multi_task():
    parser = argparse.ArgumentParser(
        description="Extracting well-formed sentences from passages")
    parser.add_argument("--data_dir", type=str, required=True, help="Names of the split files dir")
    parser.add_argument("--sub_files", type=str, required=True, help="Names of the split files, seperated by ';'")
    parser.add_argument("--nsplit", type=int, required=True, help="Num of processes to run simultaneously")
    parser.add_argument("--func_name", type=str, required=True, help="Function name applied to each sample")
    parser.add_argument("--post_action", type=str, default="merge_to_single", help="Function name applied to the post cleaning and merging phase")
    parser.add_argument("--out_filename", type=str, default=None, help="")
    parser.add_argument("--text_cols", type=str, default=None, help="")
    parser.add_argument("--use_gpu", action="store_true", help="whether use torch.multiprocessing for multi-thread")
    parser.add_argument("--no_sort", action="store_false", help="for pred_res selection or div measure only")
    parser.add_argument("--keep_col", type=str, default=None, help="")
    args = parser.parse_args()
    # # split_inspect()
    # main()
    if not args.use_gpu:
        run_split(args)
    else:
        run_gpu_split(args)


func_name_mapping = {
    "mono_sent_split": mono_sent_split_file,
    "fix_text_only": fix_text_only, 
    "mono_sent_split_text_file": mono_sent_split_text_file,
    "entity_mask": entity_mask,
    "backtrans_filter": backtrans_filter,
    "entity_mask_align": entity_mask_align,
    "diversity_measure": diversity_measure,
    "compute_tree_edit": compute_tree_edit,
    "pred_res_select": pred_res_select,
    "extract_editable": extract_editable,
    "filter_editable": filter_editable,
}

post_action_mapping = {
    "merge_to_single": merge_to_single,
    "merge_to_multiple": merge_to_multiple,
}

if __name__ == '__main__':
    # single_task()
    multi_task()
