import itertools
import pickle
from functools import reduce
import pandas as pd
import os.path
import random

from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler

from feature_creator import FeatureCreator
from sklearn.linear_model import LogisticRegression, LinearRegression

from my_classifier import MyClassifier
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
    test_mapping_file = resource_dir + "test_mapping.csv"
    test_tp_file = resource_dir + "test_map_tp.csv"
    test_fp_file = resource_dir + "test_map_fp.csv"
    test_fn_file = resource_dir + "test_map_fn.csv"
    test_p_file = resource_dir + "test_results_map.csv"
    test_unmapped_file = resource_dir + "test_results_unmapped.csv"
    classifier_pickle_file = resource_dir + "classifier.pkl"

    tp_file = resource_dir + "map_tp.csv"
    fp_file = resource_dir + "map_fp.csv"
    fn_file = resource_dir + "map_fn.csv"

    input_examples_file = resource_dir + "examples.csv"
    input_file_partner_1 = resource_dir + "Partner1.csv"
    input_file_partner_2 = resource_dir + "Partner2.csv"




    if not os.path.isfile(classifier_pickle_file):
        examples = read_record_from_csv(input_examples_file)
        original_columns = examples.columns
        examples['key'] = examples['p1.key'] + '_' + examples['p2.key']
        examples.set_index('key', inplace=True)
        true_mapping = examples[["p1.key", "p2.key"]]

        extended_examples_by_country, source_examples_by_country = \
            create_extended_examples_from_map_by_country(examples, true_mapping)

        # classifier = LinearRegression()
        # classifier = MyClassifier()
        classifier = ensemble.GradientBoostingClassifier()
        all_examples = pd.concat(extended_examples_by_country.values())
        x_to_train = all_examples[all_examples.columns.difference(original_columns)]
        x_to_train = x_to_train[x_to_train.columns.difference(['result'])]
        y_train = all_examples['result']
        classifier.fit(x_to_train, y_train)
        pickle.dump(classifier, open(classifier_pickle_file, "wb"))

        df_full_data_and_predictions, test_true_mapping = predict_matching_and_save_to_file(classifier,
                                                                                            extended_examples_by_country,
                                                                                            test_mapping_file, original_columns)

        new_mapping = read_record_from_csv(test_mapping_file)
        accuracy, coverage, (tp_mapping, fp_mapping, fn_mapping) = calc_stats(new_mapping, test_true_mapping)
        unmapped_map = calc_map_of_all_unmapped_values(df_full_data_and_predictions, new_mapping)

        example_1, example_2 = split_p1_p2_set_from_combined_examples(examples)

        total_samples = sum(map(lambda x: x[1].shape[0], source_examples_by_country.items()))
        true_positive_combined = calc_mapping_results(example_1, example_2, tp_mapping)
        false_positive_combined = calc_mapping_results(example_1, example_2, fp_mapping)
        false_negative_combined = calc_mapping_results(example_1, example_2, fn_mapping)
        mapping_combined = calc_mapping_results(example_1, example_2, new_mapping)
        unmapped_combined = calc_mapping_results(example_1, example_2, unmapped_map)

        write_result_csv(true_positive_combined, test_tp_file)
        write_result_csv(false_positive_combined, test_fp_file)
        write_result_csv(false_negative_combined, test_fn_file)
        write_result_csv(mapping_combined, test_p_file)
        write_result_csv(unmapped_combined, test_unmapped_file)

        print("Original test, Accuracy = %f, coverage=%f, total=%d, tp=%d, fp=%d, fn=%d" %
              (accuracy, coverage, total_samples, tp_mapping.shape[0], fp_mapping.shape[0], fn_mapping.shape[0]))

    df_1 = read_record_from_csv(input_file_partner_1)
    df_2 = read_record_from_csv(input_file_partner_2)
    len = min(df_1.shape[0],df_2.shape[0])
    df_1 = df_1[1:len]
    df_2 = df_2[1:len]
    df = pd.concat([df_1,df_2], axis=1)
    df['key'] = df['p1.key'] + '_' + df['p2.key']
    df.set_index('key', inplace=True)
    original_columns = df.columns

    df['p1.country_code'] = df.apply(lambda row: str(row['p1.country_code']),axis=1)
    df['p2.country_code'] = df.apply(lambda row: str(row['p2.country_code']),axis=1)

    fict_mapping = df[["p1.key", "p2.key"]]

    loaded_model = pickle.load(open(classifier_pickle_file, "rb"))
    extended_examples_by_country, source_examples_by_country = \
        create_extended_examples_from_map_by_country(df, fict_mapping)

    df_full_data_and_predictions, test_true_mapping = predict_matching_and_save_to_file(loaded_model,
                                                                                        extended_examples_by_country,
                                                                                        mapping_file, original_columns)


    new_mapping = read_record_from_csv(mapping_file)
    unmapped_map = calc_map_of_all_unmapped_values(df_full_data_and_predictions, new_mapping)
    example_1, example_2 = split_p1_p2_set_from_combined_examples(df)

    total_samples = sum(map(lambda x: x[1].shape[0], source_examples_by_country.items()))
    mapping_combined = calc_mapping_results(example_1, example_2, new_mapping)
    unmapped_combined = calc_mapping_results(example_1, example_2, unmapped_map)

    write_result_csv(mapping_combined, mapping_file)
    write_result_csv(unmapped_combined, mapping_differnce_file)


def predict_matching_and_save_to_file(classifier, extended_examples_by_country, mapping_file, original_columns):
    df_full_data_and_predictions, predictions_index = \
        run_classifier_on_extended_samples(classifier, extended_examples_by_country, original_columns)
    test_true_mapping = df_full_data_and_predictions.loc[df_full_data_and_predictions['result'] == 1] \
        [["p1.key", "p2.key"]].set_index('p1.key')
    test_map = pd.DataFrame(columns=['key', 'p1.key', 'p2.key'])
    test_map['key'] = predictions_index
    test_map.set_index("key", inplace=True)
    split_index(test_map)
    test_map.reset_index()
    write_result_csv(test_map, mapping_file, False)
    return df_full_data_and_predictions, test_true_mapping


def calc_map_of_all_unmapped_values(df_full_data_and_predictions, new_mapping):
    all_p1_keys = set(df_full_data_and_predictions["p1.key"])
    all_p2_keys = set(df_full_data_and_predictions["p2.key"])
    all_p1_keys_mapped = set(new_mapping["p1.key"])
    all_p2_keys_mapped = set(new_mapping["p2.key"])
    unmapped_p1_keys = all_p1_keys.difference(all_p1_keys_mapped)
    unmapped_p2_keys = all_p2_keys.difference(all_p2_keys_mapped)
    unmapped_indexes = [p1 + "_" + p2 for p1 in unmapped_p1_keys for p2 in unmapped_p2_keys]
    unmapped_map = pd.DataFrame(columns=['key', 'p1.key', 'p2.key'])
    unmapped_map['key'] = unmapped_indexes
    unmapped_map.set_index("key", inplace=True)
    if unmapped_map.shape[0] > 0:
        split_index(unmapped_map)
    return unmapped_map


def run_classifer_on_examples(examples, original_columns, classifier, true_mapping):
    extended_examples_by_country, source_examples_by_country = \
        create_extended_examples_from_map_by_country(examples, true_mapping)

    df_full_data_and_predictions, predictions_index = \
        run_classifier_on_extended_samples(classifier, extended_examples_by_country, original_columns)
    return df_full_data_and_predictions, predictions_index, source_examples_by_country


def create_extended_examples_from_map_by_country(examples, true_mapping=None):
    source_examples_by_country = split_dataframe_by_country(examples)
    extended_examples_by_country = {}
    for country, country_samples in source_examples_by_country.items():  # All hotels in one country
        fc = FeatureCreator(country_samples)
        features = fc.create_permutations_add_features(True)
        if true_mapping is not None:
            x_y_test = create_features_from_data_and_mapping(features, true_mapping)
        else:
            x_y_test = features
            x_y_test['result'] = 0
        extended_examples_by_country[country] = x_y_test
    return extended_examples_by_country, source_examples_by_country


def run_classifier_on_extended_samples(classifier, extended_examples_by_country, original_columns):
    results_by_country = {}
    indexes_by_country = {}
    for country, extended_country_samples in extended_examples_by_country.items():  # All hotels in one country
        y_test_country = extended_country_samples['result']
        x_test_country = extended_country_samples[extended_country_samples.columns.difference(original_columns)]
        x_test_country = x_test_country[x_test_country.columns.difference(['result'])]
        if x_test_country.shape[0] > 0:
            country_predictions_index, country_test_results = \
                predict_pick_store(classifier, x_test_country, y_test_country)
            results_by_country[country] = country_test_results
            indexes_by_country[country] = country_predictions_index
    df_full_data_and_predictions = combine_dataframes(results_by_country.values())
    predictions_index = list(itertools.chain.from_iterable(indexes_by_country.values()))
    return df_full_data_and_predictions, predictions_index


def add_features_columns(countries, examples_train, mapping_train):
    examples_by_countries = split_dataframe_by_country(countries, examples_train)
    features = []
    # Split the samples by country, sample from each country
    features_by_country = {country: preprocess_and_create_features(country_samples, mapping_train)
                           for country, country_samples in examples_by_countries.items()}

    features_total = combine_dataframes(features_by_country.values())
    return features_total


def preprocess_and_create_features(country_samples, mapping_train):
    preprocess_data(country_samples, "p1")
    preprocess_data(country_samples, "p2")
    new_features = create_features_from_data_and_mapping(country_samples, mapping_train)
    return new_features


def combine_dataframes(features):
    features_total = reduce(lambda x, y: pd.concat([x, y]), features)
    return features_total


def predict_pick_store(classifier, x_test, y_test):
    predictions = classifier.predict(x_test)
    test_results = x_test.copy()
    test_results['result'] = y_test
    test_results['prediction'] = predictions
    split_index(test_results)
    predictions_index = pick_predictions_from_results(test_results)
    return predictions_index, test_results


def pick_predictions_from_results(test_results):
    prediction_threshold = 0.500001
    test_results_sorted = test_results.sort_values(by=['prediction'], ascending=False)
    p1_used = set([])
    p2_used = set([])
    predictions_index = []
    for index, row in test_results_sorted.iterrows():
        prediction = row['prediction']
        if prediction < prediction_threshold:
            break
        if (row['p1.key'] not in p1_used and row['p2.key']) not in p2_used:
            predictions_index.append(index)
            p1_used.add(row['p1.key'])
            p2_used.add(row['p2.key'])
    return predictions_index


def split_dataframe_by_country(examples):
    countries = sorted((examples['p1.country_code'].append(examples['p2.country_code'])).unique())
    # countries = countries[0:9]  # ASSAF
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


def create_features_from_data_and_mapping(features, true_mapping):
    one_hot_encoding = create_one_hot_from_features_and_mapping(features, true_mapping)
    features = pd.merge(features, one_hot_encoding, left_index=True, right_index=True)
    return features


if __name__ == "__main__":
    main()
