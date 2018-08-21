import math
import pandas as pd
import random
import itertools
from match_utils import read_record_from_csv,sample_dataframe,clone_empty_frame,\
    write_result_csv, calc_mapping_results,create_test_set_from_examples,calc_stats


def explore_examples_diff(df, print_values=False):
    p1_keys = [key for key in df.keys() if key.startswith("p1") and "key" not in key]
    key_pairs = [[key,key.replace("p1","p2")] for key in p1_keys]
    print("Number of examples is:", df.shape[0])
    for pair in key_pairs:
        current_columns = df.loc[:, pair]
        diff = current_columns[current_columns[pair[0]] != df[pair[1]]]
        print("Printing mismatches for keys: ",pair, " number of mismatches:",diff.shape[0])
        if print_values:
            with pd.option_context('display.max_rows', None, 'display.max_columns', 3, 'max_colwidth',100,'expand_frame_repr',False,'display.width',150):
                print(diff)
            print("-----")

def suggest_mapping(set1,set2):
    return set1["p1.key"]


def compare_mappings(mapping,true_mapping):
    tp = mapping[ mapping["p1.key"] == true_mapping[""]]


def create_random_mapping(true_mapping, percentage_size, percentage_wrong):
    wrong_pointer = true_mapping.iloc[0]["p2.key"]
    map_sample = sample_dataframe(true_mapping, percentage_size)
    random_elements = random.sample(range(1, map_sample.shape[0]), round(percentage_wrong * map_sample.shape[0]))
    map_sample["p2.key"].iloc[random_elements] = wrong_pointer
    return map_sample


def create_mapping_by_grade(set_1, set_2, grading_function, threshold):
    grades = dict()
    for index1, row1 in set_1.iterrows():
        key = row1["p1.key"]
        for index2, row2 in set_2.iterrows():
            grade = grading_function(row1,row2)
            if (key not in grades) or (grades[key][0] < grade):
                grades[key] = (grade, row2["p2.key"])

    result = pd.DataFrame(columns = ["p1.key","p2.key"])
    res = {"p1.key": [], "p2.key": []}
    for key, best in grades.items():
        if best[0] > threshold:
            res["p1.key"].append(key)
            res["p2.key"].append(best[1])

    result = pd.DataFrame.from_dict(res);
    #result.set_index("p1.key", inplace=True)
    return result;

def constant_grading_function(row1,row2):
    return 1;

def random_grading_function(row1,row2):
    return random.randint(1,10000)

def naive_grading_function(row1,row2):
    if row1["p1.hotel_name"] == row2["p2.hotel_name"]:
        return 1
    return 0


def main():
    resource_dir = "..\\Resources\\"
    mapping_file = resource_dir + "mappings.csv"
    mapping_differnce_file = resource_dir + "unmapped.csv"
    tp_file = resource_dir + "map_tp.csv"
    fp_file = resource_dir + "map_fp.csv"
    fn_file = resource_dir + "map_fn.csv"

    examples = read_record_from_csv(resource_dir + "examples.csv")
    #examples = sample_dataframe(examples,0.2)  # Work with sample for now
    # explore_examples_diff(examples, True)
    example_1, example_2 = create_test_set_from_examples(examples)
    maps = create_mapping_by_grade(example_1,example_2,naive_grading_function,0.5)
    # combined_structure, unmapped_values = calc_mapping_results(example_1,example_2,maps)


    true_mapping = examples[["p1.key", "p2.key"]]

    # mapping = create_random_mapping(true_mapping, percentage_size=0.6, percentage_wrong=0.3)
    # mapping.set_index("p1.key", inplace=True)

    write_result_csv(maps, mapping_file)
    new_mapping = read_record_from_csv(mapping_file)

    #print(new_mapping.to_string())
    accuracy, coverage, (true_positive, false_positive, false_negative) = calc_stats(new_mapping, true_mapping)
    true_positive_combined  = calc_mapping_results(example_1, example_2, true_positive)
    false_positive_combined = calc_mapping_results(example_1, example_2, false_positive)
    false_negative_combined = calc_mapping_results(example_1, example_2, false_negative)

    write_result_csv(true_positive_combined,  tp_file)
    write_result_csv(false_positive_combined, fp_file)
    write_result_csv(false_negative_combined, fn_file)


    print("Accuracy = %f, coverage=%f" % (accuracy,coverage))


if __name__ == "__main__":
    main()