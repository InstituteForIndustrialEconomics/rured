import datetime
import json
import os
import random
from glob import glob
from typing import Dict, List, Optional, Set

import fire
import razdel


def open_lines(filename: str) -> Set[str]:
    """
    readlines + set
    """
    with open(filename) as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    output = set(lines)
    return output


def check_pair_in_range(start: int, end: int, check_range: range) -> bool:
    if range(max(start, check_range.start), min(end, check_range.stop)):
        return True
    else:
        return False


def get_annotation_from_file(
    annotation_filename: str, with_relations_only: bool = False
) -> List[dict]:
    with open(annotation_filename) as f:
        annotation_str: str = f.read()
    annotation_values: List[dict] = []
    if with_relations_only and "Arg1:" not in annotation_str:
        return []
    annotation: List = annotation_str.split("\n")
    annotation = [a.strip() for a in annotation]
    annotation = [a for a in annotation if a]
    annotation = [a.split("\t") for a in annotation]
    annotation_entries: dict = dict()
    ners = [a for a in annotation if a[0][0] == "T"]
    relations = [a for a in annotation if a[0][0] == "R"]
    annotation = ners + relations
    for a in annotation:
        annotation_entry: dict = dict()
        if a[0].startswith("T"):
            annotation_type = "NER"
        elif a[0].startswith("R"):
            annotation_type = "Relation"
        else:
            annotation_type = "Unknown"
        annotation_entry["type"] = annotation_type
        annotation_entry["code"] = annotation_code = a[0]

        annotation_body = a[1]
        annotation_body = annotation_body.replace("Arg1:", "").replace(
            "Arg2:", ""
        )
        annotation_body = annotation_body.split()
        annotation_entry["ann_type"] = annotation_body[0]
        if annotation_type == "NER":
            if len(a) <= 2:
                continue
            annotation_entry["start"] = int(annotation_body[1])
            annotation_entry["end"] = int(annotation_body[2])
            annotation_entry["text"] = a[2]
        else:
            for item_id, _ in enumerate(("subj", "obj")):
                item_id += 1
                for position in ("start", "end"):
                    annotation_entry[
                        f"{item_id}_{position}"
                    ] = annotation_entries[annotation_body[item_id]][position]
            annotation_entry["subj_ref"] = annotation_body[1]
            annotation_entry["obj_ref"] = annotation_body[2]
        annotation_entries[annotation_code] = annotation_entry
    annotation_values = list(annotation_entries.values())
    return annotation_values


def get_brat_annotations(
    path="brat/data/relations/Экономика/*/*.ann", with_relations_only=False
) -> Dict[str, list]:
    """
    path is string that `glob` accepts
    """
    annotations = glob(path)

    attach_flag = False
    text_data: Dict[str, list] = dict()

    for annotation_filename in annotations:
        try:
            annotation_values = get_annotation_from_file(
                annotation_filename, with_relations_only=with_relations_only
            )
        except (KeyError, ValueError, IndexError) as ex:
            print(ex, annotation_filename)
        if with_relations_only:
            if len(annotation_values) <= 3:
                continue
        else:
            if not annotation_values:
                continue
        text_filename = annotation_filename.replace(".ann", ".txt")
        with open(text_filename) as f:
            text = f.read()
        sents = list(razdel.sentenize(text))
        prev_sent = None
        text_data[annotation_filename] = []
        for sent in sents:
            if attach_flag and prev_sent:
                sent.text = prev_sent.text + " " + sent.text
                sent.start = prev_sent.start
                attach_flag = False
                prev_sent = None
            sent_tokens = list(razdel.tokenize(sent.text))
            sent_ners = ["O" for t in sent_tokens]
            sent_brat_codes = ["O" for t in sent_tokens]
            sent_data: Dict = dict()
            sent_data["token"] = [t.text for t in sent_tokens]
            sent_data["text"] = sent.text
            sent_data["relation"] = False
            sent_data["relations"] = []
            for _, value in enumerate(annotation_values):
                annotation_code = value["code"]
                if value["type"] == "NER":
                    start = value["start"]
                    end = value["end"]
                    ner_text = value["text"]
                    ner_type = value["ann_type"]
                    overlaps = check_pair_in_range(
                        start, end, range(sent.start, sent.stop)
                    )
                    if overlaps:
                        for s_t_i, s_t in enumerate(sent_tokens):
                            token_overlaps = check_pair_in_range(
                                start,
                                end,
                                range(
                                    s_t.start + sent.start,
                                    s_t.stop + sent.start,
                                ),
                            )
                            if token_overlaps:
                                if ner_text.startswith(s_t.text):
                                    sent_ners[s_t_i] = "B-" + ner_type
                                else:
                                    sent_ners[s_t_i] = "I-" + ner_type
                                sent_brat_codes[s_t_i] = annotation_code
                if value["type"] == "Relation":
                    t1_overlaps = check_pair_in_range(
                        value["1_start"],
                        value["1_end"],
                        range(sent.start, sent.stop),
                    )
                    t2_overlaps = check_pair_in_range(
                        value["2_start"],
                        value["2_end"],
                        range(sent.start, sent.stop),
                    )
                    if t1_overlaps and t2_overlaps:
                        sent_data["relation"] = True
                        relation: Dict = dict()
                        relation["relation"] = value["ann_type"]
                        relation["code"] = annotation_code
                        for entry in ("subj", "obj"):
                            codes = [
                                i
                                for i, s in enumerate(sent_brat_codes)
                                if s == value[f"{entry}_ref"]
                            ]
                            if not codes:
                                continue
                            relation[f"{entry}_type"] = (
                                sent_ners[codes[0]]
                                .replace("B-", "")
                                .replace("I-", "")
                            )
                            relation[f"{entry}_brat_code"] = value[
                                f"{entry}_ref"
                            ]
                            relation[f"{entry}_start"] = codes[0]
                            relation[f"{entry}_end"] = codes[-1]
                        if relation:
                            sent_data["relations"].append(relation)
                    elif t1_overlaps or t2_overlaps:
                        prev_sent = sent
                        attach_flag = True
            sent_data["stanford_ner"] = sent_ners
            sent_data["brat_codes"] = sent_brat_codes
            sent_data["start"] = sent.start
            sent_data["stop"] = sent.stop
            # all_data.append(sent_data)
            text_data[annotation_filename].append(sent_data)
    return text_data


def create_ner_dataset(filter_data):
    random.shuffle(filter_data)

    train = filter_data[: int(len(filter_data) * 0.8)]
    valid = filter_data[
        int(len(filter_data) * 0.8) : int(len(filter_data) * 0.9)
    ]
    test = filter_data[int(len(filter_data) * 0.9) :]

    keys = ["train", "valid", "test"]

    datasets = dict(zip(keys, [train, valid, test]))

    for name, data in datasets.items():
        data_strings = []
        for d in data:
            zipped = list(zip(*[d["token"], d["stanford_ner"]]))
            data_string = "\n".join(["\t".join(z) for z in zipped]) + "\n"
            data_strings.append(data_string)
        if not os.path.exists("ner_data/"):
            os.makedirs("ner_data")
        with open(f"ner_data/{name}.txt", "w") as f:
            f.write("\n".join(data_strings))


def main(path: Optional[str] = None):
    """[Convert files from brat to tacred]
    
    Keyword Arguments:
        path {Optional[str]} -- [is a string in glob format] (default: {None})
         e.g. "brat/data/relations/Экономика/*/*.ann"
    """
    if path:
        if not path.endswith(".ann"):
            raise Exception("path should end with '.ann'")
        text_data = get_brat_annotations(path)
    else:
        text_data = get_brat_annotations()
    all_data: List[Dict] = []
    for k, v in text_data.items():
        for sent_i, sent in enumerate(v):
            sent["filename"] = k
        all_data += list(v)

    with open(f"all_data.json", "w") as f:
        json.dump(all_data, f)
    create_ner_dataset(all_data)


if __name__ == "__main__":
    fire.Fire(main)
