import re
from functools import reduce
import pandas as pd


class FeatureCreator:
    def __init__(self, set1, set2):
        self.set_1 = set1
        self.set_2 = set2

    hotel_name_redundant_keywords = ['Hotel', 'Inn', 'The', '&', 'and', 'Suites', 'House', 'Villa', '-', 'and', 'Lodge',
                                     'Apartments',
                                     'Hostel', 'Boutique', 'Apartment', 'Motel', 'Guest', 'Guesthouse', 'by',
                                     'Grand', 'Residence', 'Villas', 'Rooms', 'Breakfast']
    hotel_name_redundant_keywords_lower = [word.lower() for word in hotel_name_redundant_keywords]

    def create_features(self, progress=False):
        field_names = ["hotel_name", "city_name", "postal_code", "hotel_address", "country_code"]

        dicts = {key: None for key in field_names}  # Dictionary of results
        methods = {field: getattr(self, "create_features_" + field, field) for field in field_names}  # Cache method
        total = self.set_1.shape[0] * self.set_2.shape[0]
        counter = 1
        for index1, row1 in self.set_1.iterrows():
            key1 = row1["p1.key"]
            for index2, row2 in self.set_2.iterrows():
                key2 = row2["p2.key"]
                combined_key = key1 + "_" + key2
                for field_name in field_names:
                    method = methods[field_name]
                    if method:
                        result = method(combined_key, dicts[field_name], row1["p1." + field_name],
                                        row2["p2." + field_name])
                    else:
                        result = None
                    dicts[field_name] = result

                if progress:
                    print("Calculating distance %d of %d" % (counter, total), end='\r')
                counter = counter + 1

        dataframes = list(map(lambda x: pd.DataFrame.from_dict(x).set_index("key"), dicts.values()))
        combined = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), dataframes)

        return combined

    def create_features_hotel_name(self, combined_key, dic, val1, val2):
        """ dic - dictionary of key: [list of values] to be converted to DataFrame"""
        columns = ["key", "hotel_distance", "hotel_number_of_same_words"]
        dic = self.init_dict(columns, dic)
        val1 = self.remove_redundant_words(val1)
        val2 = self.remove_redundant_words(val2)
        feature_1 = self.string_distance_normalized(val1, val2)

        diffs = self.word_frequency_diffs(val1, val2)
        values = [combined_key, feature_1, diffs]
        for index, value in enumerate(columns):
            dic[value].append(values[index])
        return dic;

    def word_frequency_diffs(self, val1, val2):
        words_1 = val1.split(" ")
        words_2 = val2.split(" ")
        words = words_1.copy()
        words.extend(words_2)
        dic_words = {word: 0 for word in words}
        for word in words_1:
            dic_words[word] = dic_words[word] + 1
        for word in words_2:
            dic_words[word] = dic_words[word] - 1
        diffs = sum(dic_words.values())
        return diffs

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

    def string_distance_normalized(self, str1, str2):
        if not str1 or not str2:
            return 1
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
            name = name.replace(word, "")
        name = name.strip()
        return name

