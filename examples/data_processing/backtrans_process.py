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
from collections import defaultdict, Counter
from tqdm.auto import tqdm
from ftfy import fix_text
from pathlib import Path
from multiprocessing import Pool

# DATA_DIR = "/apdcephfs/share_916081/timxthuang/bt_files/mono_en/msmarco"
# ceph_data_dir = "/apdcephfs/share_916081/timxthuang/bt_files/mono_en/msmarco"
# spacy_model_dir = "/apdcephfs/share_916081/timxthuang/cache/"


# data_dir = "/apdcephfs/share_916081/timxthuang/bt_files/mono_en/msmarco"
# out_dir = "/apdcephfs/share_916081/timxthuang/bt_files/mono_en/msmarco_test"


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
                    valid_sent.append(sent.text.strip())
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

    ln_count = sum(1 for _ in Path(src_file).open(encoding="utf-8", errors='ignore').readlines())
    progress_bar = tqdm(range(ln_count), disable=not args.major_process)
    spacy_pipe = spacy.load("en_core_web_sm")
    spacy_pipe.add_pipe("language_detector")

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
    
    with open(high_outfile, "w", encoding="utf-8") as fOut_high,\
            open(low_outfile, "w", encoding="utf-8") as fOut_low,\
            open(trash_outfile, "w", encoding="utf-8") as fOut_trash:
        for idx in range(ln_count):
            # linecache starts at 1
            idx += 1
            ln_text = linecache.getline(str(src_file), idx).rstrip("\n")
            # get sents
            sent1_text, sent2_text = json.loads(ln_text).get("src_en"), json.loads(ln_text).get("pred_en")
            progress_bar.update(1)
            if not sent1_text or not sent2_text:
                fOut_trash.write(ln_text)
                fOut_trash.write("\n")
                continue                
            sent1 = sent1_text.strip().lower().split(" ")
            sent2 = sent2_text.strip().lower().split(" ")
            # measure div and other heuristic rules (labeled as tag)
            #   tag: high-edit; low-edit; clean (length short/long or misalighn, `< unk >` issue 
            if not_valid(sent1, sent2):
                fOut_trash.write(ln_text)
                fOut_trash.write("\n")
                continue
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
    progress_bar = tqdm(range(ln_count), disable=not args.major_process)
    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx in range(ln_count):
            # linecache starts at 1
            idx += 1
            line = linecache.getline(str(in_file), idx).rstrip("\n")
            content = json.loads(line)
            ln_text = content.get("text")
            clean_text = fix_text(ln_text)
            progress_bar.update(1)
            content["text"] = clean_text
            if "\\u" in ln_text:
                print(f"{ln_text} is still dirty")
                continue
            json.dump(
                content,
                f_out
            )
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
    parser.add_argument("--out_filename", type=str, default=None, help="Target processing file")
    # parser.add_argument("--filename1", type=str, default=None, help="Target processing file1")
    # parser.add_argument("--filename2", type=str, default=None, help="Target processing file2")
    args = parser.parse_args()
    # # split_inspect()
    # main()
    run_split(args)


def fine_filter():
    parser = argparse.ArgumentParser(
        description="Extracting more fine-grained sentences from passages")
    parser.add_argument("--data_dir", type=str, required=True, help="Names of the split files dir")
    parser.add_argument("--filename", type=str, required=True, help="Raw filename")
    parser.add_argument("--tgt_num", type=int, default=50000, help="Raw filename")
    parser.add_argument("--out_filename", type=str, default=None, help="Raw filename")
    args = parser.parse_args()

    spacy_pipe = spacy.load("en_core_web_trf")
    if "." not in args.filename:
        name_prefix = args.filename
        args.filename = f"{args.filename}.jsonl"
    else:
        name_prefix, _ = args.filename.split(".")
    with open(os.path.join(args.data_dir, args.filename), encoding="utf-8") as f_in:
        all_sents = f_in.readlines()
    
    sent_idx = list(range(len(all_sents)))
    random.shuffle(sent_idx)
    
    def strict_filter(ln_text):
        doc = spacy_pipe(ln_text)
        valid_len = sum([token.pos_ != "PUNCT" and token.pos_ != "NUM" for token in doc])
        if valid_len < 20:
            return False
        if not doc[0].is_title:
            return False
        return True

    progress_bar = tqdm(range(args.tgt_num))
    out_filename = args.out_filename if args.out_filename is not None else f"{name_prefix}-strict.jsonl"
    out_file = os.path.join(args.data_dir, out_filename)
    valid_num = 0
    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx in sent_idx:
            line = all_sents[idx]
            content = json.loads(line)
            ln_text = content.get("text")
            if strict_filter(ln_text):
                json.dump(
                    content,
                    f_out,
                    ensure_ascii=False,
                )
                f_out.write("\n")
                progress_bar.update(1)
                valid_num += 1
                if valid_num > args.tgt_num:
                    break
            else:
                continue    


func_name_mapping = {
    "mono_sent_split": mono_sent_split_file,
    "fix_text_only": fix_text_only, 
    "mono_sent_split_text_file": mono_sent_split_text_file,
    "entity_mask": entity_mask,
    "backtrans_filter": backtrans_filter,
    "entity_mask_align": entity_mask_align,
}

post_action_mapping = {
    "merge_to_single": merge_to_single,
    "merge_to_multiple": merge_to_multiple,
}

if __name__ == '__main__':
    # single_task()
    multi_task()
    # fine_filter()
