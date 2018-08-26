import re
from functools import reduce
import pandas as pd

from match_utils import split_p1_p2_set_from_combined_examples


class FeatureCreator:
    def __init__(self, examples):
        self.set_1, self.set_2 = split_p1_p2_set_from_combined_examples(examples)


    hotel_name_redundant_keywords = ['Hotel', 'Inn', 'The', ',', '-', '&', 'and', 'Suites', 'House', 'Villa', '-',
                                     'and', 'Lodge', 'Branch',
                                     'Apartments', 'Luxury', 'Restaurant',
                                     'Hostel', 'Boutique', 'Apartment', 'Motel', 'Guest', 'Guesthouse', 'by',
                                     'Grand', 'Residence', 'Villas', 'Rooms', 'Breakfast']
    hotel_name_redundant_keywords_lower = [word.lower() for word in hotel_name_redundant_keywords]

    def join_multiply_frames(self, df1, df2):
        df1['tmp'] = 1
        df2['tmp'] = 1

        df = pd.merge(df1, df2, on=['tmp'])
        df = df.drop('tmp', axis=1)
        return df

    field_names = ["hotel_name", "city_name", "postal_code", "hotel_address", "country_code"]

    def create_permutations_add_features(self, progress=False):
        joined_frame = self.create_cartesian_set(self.set_1, self.set_2)
        combined = self.add_features_to_dataset(joined_frame, progress)
        return combined

    def create_cartesian_set(self, set_1,set_2):
        joined_frame = self.join_multiply_frames(set_1, set_2)
        joined_frame['key'] = joined_frame['p1.key'] + "_" + joined_frame['p2.key']
        joined_frame.set_index('key', inplace=True)
        return joined_frame

    def add_features_to_dataset(self, dataset, progress=False):
        """
        :param dataset: Data
        :param progress: Print Progress
        :return: Features with result=1 or result=0
        """
        total = dataset.shape[0]
        methods = {field: getattr(self, "create_features_" + field, field) for field in
                   self.field_names}  # Cache method
        counter = 1
        new_features = pd.DataFrame(columns=['key'])
        new_features.set_index('key', inplace=True)
        dicts = {key: None for key in self.field_names}  # Dictionary of results
        for index, row in dataset.iterrows():
            combined_key = index
            for field_name in self.field_names:
                method = methods[field_name]
                if method:
                    result = method(combined_key, dicts[field_name], row["p1." + field_name],
                                    row["p2." + field_name])
                else:
                    result = None
                dicts[field_name] = result
                if progress:
                    print("  Calculating distance %d of %d" % (counter, total), end='\r')
            counter = counter + 1
        print()
        dataframes = list(map(lambda x: pd.DataFrame.from_dict(x).set_index("key"), dicts.values()))

        # All created features in one dataframe
        combined = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), dataframes)
        # Store in the new DataFrame
        combined = new_features.append(combined)
        combined = pd.merge(dataset, combined, left_index=True, right_index=True)
        return combined

    def create_features_hotel_name(self, combined_key, dic, val1, val2):
        """ dic - dictionary of key: [list of values] to be converted to DataFrame"""
        columns = ["key", "hotel_name_difference"]
        dic = self.init_dict(columns, dic)
        val1 = self.remove_redundant_words(val1)
        val2 = self.remove_redundant_words(val2)
        letter_distance = self.string_distance_normalized(val1, val2)
        words_different = self.normalized_word_frequency_diffs(val1, val2)
        combined_difference = min(letter_distance, words_different)
        values = [combined_key, combined_difference]
        for index, value in enumerate(columns):
            dic[value].append(values[index])
        return dic;

    def normalized_word_frequency_diffs(self, val1, val2):
        words_1 = val1.split(" ")
        words_2 = val2.split(" ")
        max_length = max(len(words_1), len(words_2))
        words = words_1.copy()
        words.extend(words_2)  # All words in both fields
        dic_words = {word: 0 for word in words}
        for word in words_1:
            dic_words[word] = dic_words[word] + 1
        for word in words_2:
            dic_words[word] = dic_words[word] - 1
        diffs = sum(map(lambda x: abs(x), dic_words.values()))
        return diffs / (0.0 + max_length)  # Normalized difference

    def create_features_country_code(self, combined_key, dic, val1, val2):
        """ dic - dictionary of key: [list of values] to be converted to DataFrame"""
        columns = ["key", "country_code_same"]
        dic = self.init_dict(columns, dic)

        res = 10
        if not val1 or not val2:
            res = 0
        if val1 == val2:
            res = 0

        values = [combined_key, res]
        for index, value in enumerate(columns):
            dic[value].append(values[index])
        return dic

    def create_features_city_name(self, combined_key, dic, val1, val2):
        columns = ["key", "city_name_distance"]
        dic = self.init_dict(columns, dic)
        feature_1 = self.string_distance_normalized(val1, val2)
        values = [combined_key, feature_1]
        for index, value in enumerate(columns):
            dic[value].append(values[index])
        return dic;

    def create_features_postal_code(self, combined_key, dic, val1, val2):
        columns = ["key", "postal_code_distance"]
        dic = self.init_dict(columns, dic)
        if val1 is None or val2 is None:
            feature_1 = 0
        else:
            feature_1 = self.string_distance_normalized(val1, val2)
        values = [combined_key, feature_1]
        for index, value in enumerate(columns):
            dic[value].append(values[index])
        return dic;

    def create_features_hotel_address(self, combined_key, dic, val1, val2):
        columns = ["key", "hotel_address_distance"]
        dic = self.init_dict(columns, dic)
        feature_1 = self.string_distance_normalized(val1, val2)
        values = [combined_key, feature_1]
        for index, value in enumerate(columns):
            dic[value].append(values[index])
        return dic;

    @staticmethod
    def iterative_levenshtein(s, t):
        """
            iterative_levenshtein(s, t) -> ldist
            ldist is the Levenshtein distance between the strings
            s and t.
            For all i and j, dist[i,j] will contain the Levenshtein
            distance between the first i characters of s and the
            first j characters of t
        """
        rows = len(s) + 1
        cols = len(t) + 1
        dist = [[0 for x in range(cols)] for x in range(rows)]
        # source prefixes can be transformed into empty strings
        # by deletions:
        for i in range(1, rows):
            dist[i][0] = i
        # target prefixes can be created from an empty source string
        # by inserting the characters
        for i in range(1, cols):
            dist[0][i] = i

        for col in range(1, cols):
            for row in range(1, rows):
                if s[row - 1] == t[col - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                     dist[row][col - 1] + 1,  # insertion
                                     dist[row - 1][col - 1] + cost)  # substitution
        return dist[rows - 1][cols - 1]

    def string_distance_normalized(self, st1, st2):
        str1 = st1
        str2 = st2

        if str1 is None:
            return 1
        if str2 is None:
            return 1
        if not isinstance(str1, str):
            str1 = str(str1)
        if not isinstance(str2, str):
            str2 = str(str2)
        if str1.lower() == str2.lower():
            return 0
        l = max(len(str1), len(str2))
        return self.iterative_levenshtein(str1.lower(), str2.lower()) / l;

    def init_dict(self, columns, dic):
        if dic is None:
            dic = {}
        if not dic:
            dic = {key: [] for key in columns}
        return dic

    def remove_redundant_words(self, name):
        for word in self.hotel_name_redundant_keywords_lower:
            name = name.replace(word, " ")
            name = name.replace("  ", " ")
        name = name.strip()
        return name
