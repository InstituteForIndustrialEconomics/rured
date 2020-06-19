import fire
import json
import itertools
import random
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


def print_relations_stats(examples):
    relations = Counter([x['relation'] for x in examples])
    print(relations.most_common())


def create(source: str = 'brat_tacred.json', dest_dir: str = 'brat_tacred_format'):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    data = json.load(open(source))
    examples = []
    n_pos_examples, n_neg_examples = 0, 0
    positions = [0, 1]
    # subj_ners = ['GROUP', 'PERSON', 'ORGANIZATION', 'CITY', 'COUNTRY', 'GPE', 'PROFESSION', 'REGION']
    data = [example for example in data if
            all([subj_key in relation for subj_key in ['subj_type', 'obj_type'] for relation in example['relations']])]
    subj_ners = set([rel['subj_type'] for relations in data for rel in relations['relations']])
    for example in data:
        pair_starts = set()
        for relation in example['relations']:
            new_example = {}
            try:
                new_example['token'] = example['token']
                new_example['relation'] = relation['relation']
                new_example['subj_start'] = relation['subj_start']
                new_example['subj_end'] = relation['subj_end']
                new_example['obj_start'] = relation['obj_start']
                new_example['obj_end'] = relation['obj_end']
                new_example['subj_type'] = relation['subj_type']
                new_example['obj_type'] = relation['obj_type']
                new_example['stanford_ner'] = example['stanford_ner']
                new_example['id'] = None
            except KeyError:
                continue
            examples.append(new_example)
            n_pos_examples += 1
            pair_starts.add(tuple(sorted([relation['subj_start'], relation['obj_start']])))

        entities = [i for i, x in enumerate(example['stanford_ner']) if x.startswith('B-')]
        for pair in itertools.product(entities, repeat=2):
            pair = tuple(sorted(pair))
            if pair[0] == pair[1] or pair in pair_starts:
                continue
            if all([example['stanford_ner'][pair[i]][2:] not in subj_ners for i in positions]):
                continue
            pair_starts.add(pair)
            random.shuffle(positions)
            if example['stanford_ner'][pair[positions[0]]][2:] not in subj_ners:
                positions = positions[::-1]

            new_example = {}
            subj_end = pair[positions[0]] + 1
            while subj_end < len(example['stanford_ner']) and example['stanford_ner'][subj_end].startswith('I-'):
                subj_end += 1
            obj_end = pair[positions[1]] + 1
            while obj_end < len(example['stanford_ner']) and example['stanford_ner'][obj_end].startswith('I-'):
                obj_end += 1
            new_example['token'] = example['token']
            new_example['relation'] = 'no_relation'
            new_example['subj_start'] = pair[positions[0]]
            new_example['subj_end'] = subj_end - 1
            new_example['obj_start'] = pair[positions[1]]
            new_example['obj_end'] = obj_end - 1
            new_example['subj_type'] = example['stanford_ner'][pair[positions[0]]][2:]
            new_example['obj_type'] = example['stanford_ner'][pair[positions[1]]][2:]
            new_example['stanford_ner'] = example['stanford_ner']
            new_example['id'] = None
            examples.append(new_example)
            n_neg_examples += 1

    print(f'number of positive examples: {n_pos_examples}')
    print(f'number of negative examples: {n_neg_examples}')
    print(f'percentage of negative examples: {n_neg_examples * 100.0 / (n_pos_examples + n_neg_examples)}%')

    train_examples, test_examples = train_test_split(range(len(examples)), test_size=0.25, shuffle=True,
                                                     random_state=42)
    test_examples, dev_examples = train_test_split(test_examples, test_size=0.5, shuffle=True, random_state=42)
    examples = np.array(examples)

    train_examples = list(examples[train_examples])
    test_examples = list(examples[test_examples])
    dev_examples = list(examples[dev_examples])

    print_relations_stats(train_examples)
    print_relations_stats(test_examples)
    print_relations_stats(dev_examples)

    json.dump(train_examples, open(os.path.join(dest_dir, 'train.json'), 'w'))
    json.dump(test_examples, open(os.path.join(dest_dir, 'test.json'), 'w'))
    json.dump(dev_examples, open(os.path.join(dest_dir, 'dev.json'), 'w'))


if __name__ == "__main__":
    fire.Fire(create)
