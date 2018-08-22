import math
from typing import List, Any, Union
from functools import reduce
import pandas as pd
import random
import itertools
from match_utils import read_record_from_csv, sample_dataframe, clone_empty_frame, \
    write_result_csv, calc_mapping_results, create_test_set_from_examples, calc_stats, iterative_levenshtein


class FeatureCreator:
    def __init__(self, key1, key2, row_1, row_2):
        self.combined_key = key1 + "_" + key2
        self.row1 = row_1
        self.row2 = row_2

    def create_features_hotel_name(self, dic, val1, val2):
        """ dic - dictionary of key: [list of values] to be converted to DataFrame"""
        my_keys = ["key", "hotel_distance", "hotel_length"]
        if dic is None:
            dic = {}
        if not dic:
            dic = {key: [] for key in my_keys}
        feature_1 = string_distance(val1, val2)
        feature_2 = max(len(val1), len(val2))
        values = [self.combined_key, feature_1, feature_2]
        for index, value in enumerate(my_keys):
            dic[value].append(values[index])
        return dic;

    def create_features_city_name(self, dic, val1, val2):
        my_keys = ["key", "city_name_distance", "city_name_length"]
        if dic is None:
            dic = {}
        if not dic:
            dic = {key: [] for key in my_keys}
        feature_1 = string_distance(val1, val2)
        feature_2 = max(len(val1), len(val2))
        values = [self.combined_key, feature_1, feature_2]
        for index, value in enumerate(my_keys):
            dic[value].append(values[index])
        return dic;


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


def new_grading_function(row1, row2, dict=None):
    # return string_distance(row1["p1.hotel_name"], row2["p2.hotel_name"])
    # features = create_features_hotel_name(None, row1["p1.hotel_name"], row2["p2.hotel_name"])
    return calculate_new_features(row1, row2, "hotel_name")
    return features["distance"][0]


# def calculate_new_features(row1, row2, field_name):
#     fc = FeatureCreator(row1["p1.key"], row2["p2.key"])
#     method = getattr(fc, "create_features_" + field_name, None)
#     if method:
#         dict = {}
#         result = method(dict, row1["p1." + field_name], row2["p2." + field_name])
#     else:
#         return None
#     return result


def create_features(set_1, set_2, progress=False):
    field_names = ["hotel_name","city_name"]

    dicts = {key: None for key in field_names}
    total = set_1.shape[0] * set_2.shape[0]
    counter = 1
    for index1, row1 in set_1.iterrows():
        key1 = row1["p1.key"]
        for index2, row2 in set_2.iterrows():
            key2 = row2["p2.key"]
            fc = FeatureCreator(key1, key2, row1, row2)
            for field_name in field_names:
                method = getattr(fc, "create_features_" + field_name, dicts[field_name])
                if method:
                    result = method(dicts[field_name], row1["p1." + field_name], row2["p2." + field_name])
                else:
                    result = None
                dicts[field_name] = result

            if progress:
                print("Calculating distance %d of %d" % (counter, total), end='\r')
            counter = counter + 1

    dataframes = list(map(lambda x: pd.DataFrame.from_dict(x).set_index("key"), dicts.values()))
    combined = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), dataframes)
    return combined




fields = ["hotel_name", "city_name", "country_code", "hotel_address", "star_rating", "postal_code"]


def string_distance(str1, str2):
    if str1.lower() == str2.lower():
        return 0
    l = max(len(str1), len(str2))
    return iterative_levenshtein(str1.lower(), str2.lower()) / l;


def create_one_hot_from_features_and_mapping(features, mapping):
    column = pd.DataFrame(features.index)
    column['result'] = 0;
    true_values = mapping['p1.key']+ "_" + mapping['p2.key']
    column.loc[column['key'].isin(true_values.values), 'result'] = 1
    column.set_index('key',inplace=True)
    return column;


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

    # maps = create_mapping_by_grade(example_1, example_2, new_grading_function, 0.3, True)
    #write_result_csv(maps, mapping_file)

    features = create_features(example_1, example_2, True)
    one_hot_encoding = create_one_hot_from_features_and_mapping(features,true_mapping)
    features = pd.merge(features, one_hot_encoding, left_index=True, right_index=True)



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
