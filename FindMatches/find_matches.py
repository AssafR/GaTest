import itertools
from functools import reduce
import pandas as pd
import random

from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler

from feature_creator import FeatureCreator
from sklearn.linear_model import LogisticRegression, LinearRegression
from pre_processor import preprocess_data
from sklearn.model_selection import train_test_split

from match_utils import read_record_from_csv, sample_dataframe, clone_empty_frame, \
    write_result_csv, calc_mapping_results, split_p1_p2_set_from_combined_examples, calc_stats, \
    add_item_to_dictionary_of_lists

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


def create_test_train(features, percentage_test):
    """ Split to train and test, y is the 'result' column"""
    data = features.loc[:, features.columns != 'result']
    target = features.loc[:, 'result']
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=percentage_test, random_state=0)
    return x_train, x_test, y_train, y_test


def split_index(features):
    # Re-Split the index in a DataFrame
    features['p1.key'] = features.apply(lambda row: row.name.split('_')[0], axis=1)
    features['p2.key'] = features.apply(lambda row: row.name.split('_')[1], axis=1)


def main():
    resource_dir = "..\\Resources\\"
    mapping_file = resource_dir + "mappings.csv"
    mapping_differnce_file = resource_dir + "unmapped.csv"
    tp_file = resource_dir + "map_tp.csv"
    fp_file = resource_dir + "map_fp.csv"
    fn_file = resource_dir + "map_fn.csv"

    examples = read_record_from_csv(resource_dir + "examples.csv")
    examples['key'] = examples['p1.key'] + "_" + examples['p2.key']
    examples.set_index('key', inplace=True)
    true_mapping = examples[["p1.key", "p2.key"]]

    sample_ratio = 0.5
    mapping_test = true_mapping.copy()
    mapping_train = mapping_test.sample(frac=sample_ratio, random_state=200)
    mapping_test = mapping_test.drop(mapping_train.index)

    countries = sorted((examples['p1.country_code'].append(examples['p2.country_code'])).unique())
    examples_by_countries = split_dataframe_by_country(countries, examples)

    print("Sampling ", sample_ratio, " of original data")

    example_left_by_country = {}
    example_right_by_country = {}
    features = []
    # Split the samples by country, sample from each country
    for country, country_samples in examples_by_countries.items():
        left_row, right_row = sample_and_split(country_samples, sample_ratio)
        preprocess_data(left_row, "p1")
        preprocess_data(right_row, "p2")
        if left_row is not None and right_row is not None:
            new_features = create_features_from_data_and_mapping(left_row, right_row, true_mapping)
            new_features['p1.country_code'] = country   # For later split
            new_features['p2.country_code'] = country
            features.append(new_features)
            add_item_to_dictionary_of_lists(example_left_by_country, country, left_row)
            add_item_to_dictionary_of_lists(example_right_by_country, country, right_row)

    features_total = combine_dataframes(features)
    percentage_test = 0.99
    print("Split, test percentage is: ", percentage_test)
    x_train, x_test, y_train, y_test = create_test_train(features_total, percentage_test)
    x_to_train = x_train[x_train.columns.difference(['p2.country_code', 'country', 'p1.country_code'])]
    x_to_test = x_test[x_test.columns.difference(['p2.country_code', 'country', 'p1.country_code'])]

    # scaler = MinMaxScaler()
    # scaler.fit(x_to_train)
    # x_test[x_test.columns] = scaler.fit_transform(x_test[x_test.columns])
    x_y_test = x_test.copy()
    x_y_test['result'] = y_test


    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)

    classifier = LogisticRegression()
    #classifier = ensemble.GradientBoostingClassifier()
    classifier.fit(x_to_train, y_train)

    x_y_test_by_country = split_dataframe_by_country(countries, x_y_test)
    results_by_country = {}
    indexes_by_country = {}
    for country, country_samples in x_y_test_by_country.items():
        print("Predict for country: ", country)
        x_test_country = country_samples[
            country_samples.columns.difference(['p1.country_code', 'p2.country_code', 'result'])]
        y_test_country = country_samples['result']
        if x_test_country.shape[0] > 0:
            country_predictions_index, country_test_results = predict_pick_store(classifier, x_test_country,
                                                                                 y_test_country)
            results_by_country[country] = country_test_results
            indexes_by_country[country] = country_predictions_index

    total_predictions = combine_dataframes(results_by_country.values())
    predictions_index = list(itertools.chain.from_iterable(indexes_by_country.values()))

    # test_results_and_original = pd.concat(
    #     [examples.loc[x_test.index], test_results, features.loc[x_test.index]], axis=1)

    # features['p1.key'] = features.apply(lambda row: row.name.split('_')[0], axis=1).set_index('p1.key')  # Can probably be a little more efficient

    # prediction = features[['p1.key','p2.key']]

    test_true_mapping = total_predictions.loc[total_predictions['result'] == 1][["p1.key", "p2.key"]].set_index(
        'p1.key')
    test_map = pd.DataFrame(columns=['key', 'p1.key', 'p2.key'])
    test_map['key'] = predictions_index
    test_map.set_index("key", inplace=True)
    split_index(test_map)
    test_map.reset_index()

    # test_map = examples.loc[predictions_index][["p1.key", "p2.key"]].set_index('p1.key')

    # predictions = [features.loc[features['p1.key'] == key]['prediction'].idxmax().split("_") for key in p1_keys]

    # maps = create_mapping_by_grade(example_1, example_2, new_grading_function, 0.3, True)
    # write_result_csv(maps, mapping_file)
    write_result_csv(test_map, mapping_file, False)

    # combined_structure, unmapped_values = calc_mapping_results(example_1,example_2,maps)

    # mapping = create_random_mapping(true_mapping, percentage_size=0.6, percentage_wrong=0.3)
    # mapping.set_index("p1.key", inplace=True)

    new_mapping = read_record_from_csv(mapping_file)
    accuracy, coverage, (true_positive, false_positive, false_negative) = calc_stats(new_mapping, test_true_mapping)

    example_1, example_2 = split_p1_p2_set_from_combined_examples(examples)

    true_positive_combined  = calc_mapping_results(example_1, example_2, true_positive)
    false_positive_combined = calc_mapping_results(example_1, example_2, false_positive)
    false_negative_combined = calc_mapping_results(example_1, example_2, false_negative)

    write_result_csv(true_positive_combined, tp_file)
    write_result_csv(false_positive_combined, fp_file)
    write_result_csv(false_negative_combined, fn_file)

    print("Accuracy = %f, coverage=%f" % (accuracy, coverage))


def combine_dataframes(features):
    features_total = reduce(lambda x, y: pd.concat([x, y]), features)
    return features_total


def predict_pick_store(classifier, x_test, y_test):
    predictions = classifier.predict(x_test)
    test_results = x_test.copy()
    test_results['result'] = y_test
    test_results['prediction'] = predictions
    split_index(test_results)
    p1_keys = test_results['p1.key'].unique()
    predictions_index = [test_results.loc[test_results['p1.key'] == key]['prediction'].idxmax() for key in
                         p1_keys]  # Pick up one mapping by "confidence"
    return predictions_index, test_results


def split_dataframe_by_country(countries, examples):
    examples_by_countries = {key:
                                 examples.loc[
                                     (examples['p1.country_code'] == key) & (examples['p2.country_code'] == key)]
                             for key in countries}
    return examples_by_countries


def sample_and_split(examples, sample_ratio):
    examples_sample = sample_dataframe(examples, sample_ratio)  # Work with sample for now
    # explore_examples_diff(examples, True)
    if not examples_sample.empty:  # Non-empty sample
        example_1, example_2 = split_p1_p2_set_from_combined_examples(examples_sample)
        return example_1, example_2
    else:
        return None, None


def create_features_from_data_and_mapping(example_1, example_2, true_mapping):
    fc = FeatureCreator(example_1, example_2)
    features = fc.create_features(True)
    one_hot_encoding = create_one_hot_from_features_and_mapping(features, true_mapping)
    features = pd.merge(features, one_hot_encoding, left_index=True, right_index=True)
    return features


if __name__ == "__main__":
    main()
