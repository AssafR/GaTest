import math
from collections import Counter
from functools import reduce

import pandas as pd
import itertools


def read_record_from_csv(filename):
    df = pd.read_csv(filename, header=0, sep=',', index_col=None)
    return df


def write_result_csv(df, filename, ind=True):
    df.to_csv(filename, index=ind)


def sample_dataframe(true_mapping, percentage_size):
    true_size = true_mapping.shape[0]
    no_samples = math.ceil(percentage_size * true_size)
    map_sample = true_mapping.sample(no_samples)
    return map_sample


def clone_empty_frame(set1):
    set_1_structure = pd.DataFrame().reindex_like(set1)
    set_1_structure = set_1_structure.iloc[0:0]
    return set_1_structure


def add_item_to_dictionary_of_lists(dic, key, item):
    if key not in dic:
        dic[key] = []
    dic[key].append(item)


def calc_mapping_results(set1, set2, mapping):
    """ Mapping is the same format as mapping.csv"""

    set_1_structure = clone_empty_frame(set1)
    set_2_structure = clone_empty_frame(set2)
    combined_structure = pd.concat([set_1_structure, set_2_structure], axis=1, join_axes=[set_1_structure.index])

    set_1 = set1.copy()
    set_2 = set2.copy()
    set_1.set_index("p1.key", inplace=True, drop=False)
    set_2.set_index("p2.key", inplace=True, drop=False)
    mapping.set_index("p1.key", inplace=True, drop=False)

    for index, row in mapping.iterrows():
        row_1 = set_1.loc[index]
        row_2 = set_2.loc[row.loc["p2.key"]]
        row_combined = row_1.append(row_2)
        combined_structure = combined_structure.append(row_combined, ignore_index=True)

    # Change order for better understanding: p1.name next to p2.name , etc.
    p1_keys = [key for key in combined_structure.keys() if key.startswith("p1")]
    new_keys = [(key, key.replace("p1", "p2")) for key in p1_keys]
    new_keys = list(itertools.chain.from_iterable(new_keys))
    combined_structure = combined_structure[new_keys]
    combined_structure.set_index("p1.key", inplace=True)

    return combined_structure


def split_p1_p2_set_from_combined_examples(df):
    p1_keys = [key for key in df.keys() if key.startswith("p1")]
    p1 = df[p1_keys]
    p2_keys = [key for key in df.keys() if key.startswith("p2")]
    p2 = df[p2_keys]
    return p1, p2


def calc_stats(mapping, true_mapping):
    """ Given two sets of mappings, return the accuracy, coverage, and difference in mapping"""

    joined_mapping = pd.merge(true_mapping, mapping, on="p1.key", how='outer', suffixes=["_true", "_suggested"])
    joined_mapping = joined_mapping[~joined_mapping['p1.key'].isnull()]

    total = mapping.shape[0]
    true_positive  = joined_mapping[joined_mapping['p2.key_true'] == joined_mapping['p2.key_suggested']]
    false_positive = joined_mapping[~joined_mapping['p2.key_suggested'].isnull()
                                    & (joined_mapping['p2.key_true'] != joined_mapping['p2.key_suggested'])]
    false_negative = joined_mapping[joined_mapping['p2.key_suggested'].isnull()]

    true_positive  = pick_and_rename_columns(true_positive,  {"p1.key": "p1.key", "p2.key_suggested": "p2.key"})
    false_positive = pick_and_rename_columns(false_positive, {"p1.key": "p1.key", "p2.key_suggested": "p2.key"})
    false_negative = pick_and_rename_columns(false_negative, {"p1.key": "p1.key", "p2.key_true": "p2.key"})

    accuracy = true_positive.shape[0] / (0.0 + total)
    coverage = true_positive.shape[0] / (0.0 + true_mapping.shape[0])

    return accuracy, coverage, (true_positive, false_positive, false_negative)


def pick_and_rename_columns(df, map_columns):
    df = df[sorted([*map_columns])]  # Sort so that all p1 are before p2
    df = df.rename(columns=map_columns)
    df.set_index("p1.key", inplace=True, drop=False)
    return df


def get_words_from_field_in_file(filename, field):
    examples = read_record_from_csv(filename)
    words = list(map(lambda x: x.split(' '), examples[field].values))
    return words


def calc_words_stats():
    resource_dir = "C:\\Users\\Assaf\\PycharmProjects\\GaTest\\Resources\\"

    words = []
    words = words + get_words_from_field_in_file(resource_dir + "examples.csv", "p1.hotel_name")
    words = words + get_words_from_field_in_file(resource_dir + "examples.csv", "p2.hotel_name")
    words = words + get_words_from_field_in_file(resource_dir + "Partner1.csv", "p1.hotel_name")
    words = words + get_words_from_field_in_file(resource_dir + "Partner2.csv", "p2.hotel_name")

    # words1 = list(map(lambda x: x.split(' '),examples["p1.hotel_name"].values))
    # words1 = words1 + list(map(lambda x: x.split(' '),examples["p2.hotel_name"].values))
    words = reduce(lambda x, y: x + y, words)
    Counter(words).most_common(50)


def explore_examples_diff(df, print_values=False):
    p1_keys = [key for key in df.keys() if key.startswith("p1") and "key" not in key]
    key_pairs = [[key, key.replace("p1", "p2")] for key in p1_keys]
    print("Number of examples is:", df.shape[0])
    for pair in key_pairs:
        current_columns = df.loc[:, pair]
        diff = current_columns[current_columns[pair[0]] != df[pair[1]]]
        print("Printing mismatches for keys: ", pair, " number of mismatches:", diff.shape[0])
        if print_values:
            with pd.option_context('display.max_rows', None, 'display.max_columns', 3, 'max_colwidth', 100,
                                   'expand_frame_repr', False, 'display.width', 150):
                print(diff)
            print("-----")
