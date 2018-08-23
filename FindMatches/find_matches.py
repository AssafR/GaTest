import math
from typing import List, Any, Union
import pandas as pd
import random
from feature_creator import FeatureCreator
import itertools
from match_utils import read_record_from_csv, sample_dataframe, clone_empty_frame, \
    write_result_csv, calc_mapping_results, create_test_set_from_examples, calc_stats, iterative_levenshtein


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


def suggest_mapping(set1, set2):
    return set1["p1.key"]


def compare_mappings(mapping, true_mapping):
    tp = mapping[mapping["p1.key"] == true_mapping[""]]

def create_random_mapping(true_mapping, percentage_size, percentage_wrong):
    wrong_pointer = true_mapping.iloc[0]["p2.key"]
    map_sample = sample_dataframe(true_mapping, percentage_size)
    random_elements = random.sample(range(1, map_sample.shape[0]), round(percentage_wrong * map_sample.shape[0]))
    map_sample["p2.key"].iloc[random_elements] = wrong_pointer
    return map_sample


def create_mapping_by_grade(set_1, set_2, grading_function, threshold, progress=False):
    grades = dict()
    total = set_1.shape[0] * set_2.shape[0]
    counter = 1
    for index1, row1 in set_1.iterrows():
        key = row1["p1.key"]
        for index2, row2 in set_2.iterrows():
            grade = grading_function(row1, row2)
            if (key not in grades) or (grades[key][0] > grade):
                grades[key] = (grade, row2["p2.key"])
            if progress:
                print("Calculating distance %d of %d" % (counter, total), end='\r')
            counter = counter + 1

    result = pd.DataFrame(columns=["p1.key", "p2.key"])
    res = {"p1.key": [], "p2.key": []}
    for key, best in grades.items():
        if best[0] < threshold:
            res["p1.key"].append(key)
            res["p2.key"].append(best[1])

    result = pd.DataFrame.from_dict(res);
    # result.set_index("p1.key", inplace=True)
    return result;


def constant_grading_function(row1, row2):
    return 1;


def random_grading_function(row1, row2):
    return random.randint(1, 10000)


def naive_grading_function(row1, row2):
    if row1["p1.hotel_name"] == row2["p2.hotel_name"]:
        return 0
    return 1


fields = ["hotel_name", "city_name", "country_code", "hotel_address", "star_rating", "postal_code"]


def create_one_hot_from_features_and_mapping(features, mapping):
    column = pd.DataFrame(features.index)
    column['result'] = 0;
    true_values = mapping['p1.key'] + "_" + mapping['p2.key']
    column.loc[column['key'].isin(true_values.values), 'result'] = 1
    column.set_index('key', inplace=True)
    return column;


def split_combined_key(row):
    split = row.name.split("_")
    row['p1.key'] = split[0]
    row['p2.key'] = split[1]


def main():
    resource_dir = "..\\Resources\\"
    mapping_file = resource_dir + "mappings.csv"
    mapping_differnce_file = resource_dir + "unmapped.csv"
    tp_file = resource_dir + "map_tp.csv"
    fp_file = resource_dir + "map_fp.csv"
    fn_file = resource_dir + "map_fn.csv"

    examples = read_record_from_csv(resource_dir + "examples.csv")
    examples = sample_dataframe(examples, 0.1)  # Work with sample for now
    # explore_examples_diff(examples, True)
    example_1, example_2 = create_test_set_from_examples(examples)
    true_mapping = examples[["p1.key", "p2.key"]]

    # Add: Preprocess the examples (lower, remove extra spaces, etc).
    # Add: If ZIP code is None, distance 0
    # Add: If country different, distance BIG
    # Add: Distance in star rating?
    # Add: Preprocess everything to a new vector of features...
    #  Features: same country, same ZIP, edit distances.
    #  Go over text in real no FP examples

    fc =
    features = create_features(example_1, example_2, True)
    one_hot_encoding = create_one_hot_from_features_and_mapping(features, true_mapping)
    features = pd.merge(features, one_hot_encoding, left_index=True, right_index=True)

    # Predict on all values except keys
    features['prediction'] = features['result']  # For test

    # Re-Split the index
    features['p1.key'] = features.apply(lambda row: row.name.split('_')[0], axis=1)
    features['p2.key'] = features.apply(lambda row: row.name.split('_')[1], axis=1)

    # features['p1.key'] = features.apply(lambda row: row.name.split('_')[0], axis=1).set_index('p1.key')  # Can probably be a little more efficient

    # prediction = features[['p1.key','p2.key']]
    p1_keys = features['p1.key'].unique()
    predictions = [features.loc[features['p1.key'] == key]['prediction'].idxmax() for key in p1_keys]
    map = features.loc[predictions][["p1.key", "p2.key"]].set_index('p1.key')
    # predictions = [features.loc[features['p1.key'] == key]['prediction'].idxmax().split("_") for key in p1_keys]

    # maps = create_mapping_by_grade(example_1, example_2, new_grading_function, 0.3, True)
    # write_result_csv(maps, mapping_file)
    write_result_csv(map, mapping_file)

    # combined_structure, unmapped_values = calc_mapping_results(example_1,example_2,maps)

    # mapping = create_random_mapping(true_mapping, percentage_size=0.6, percentage_wrong=0.3)
    # mapping.set_index("p1.key", inplace=True)

    new_mapping = read_record_from_csv(mapping_file)

    # print(new_mapping.to_string())
    accuracy, coverage, (true_positive, false_positive, false_negative) = calc_stats(new_mapping, true_mapping)
    true_positive_combined = calc_mapping_results(example_1, example_2, true_positive)
    false_positive_combined = calc_mapping_results(example_1, example_2, false_positive)
    false_negative_combined = calc_mapping_results(example_1, example_2, false_negative)

    write_result_csv(true_positive_combined, tp_file)
    write_result_csv(false_positive_combined, fp_file)
    write_result_csv(false_negative_combined, fn_file)

    print("Accuracy = %f, coverage=%f" % (accuracy, coverage))


if __name__ == "__main__":
    main()
