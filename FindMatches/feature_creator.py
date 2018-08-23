class FeatureCreator:
    def __init__(self, key1, key2, row_1, row_2):
        self.combined_key = key1 + "_" + key2
        self.row1 = row_1
        self.row2 = row_2

    def create_features(set_1, set_2, progress=False):
        field_names = ["hotel_name", "city_name"]

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

    def create_features_hotel_name(self, dic, val1, val2):
        """ dic - dictionary of key: [list of values] to be converted to DataFrame"""
        my_keys = ["key", "hotel_distance", "hotel_length"]
        if dic is None:
            dic = {}
        if not dic:
            dic = {key: [] for key in my_keys}
        feature_1 = self.string_distance(val1, val2)
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
        feature_1 = self.string_distance(val1, val2)
        feature_2 = max(len(val1), len(val2))
        values = [self.combined_key, feature_1, feature_2]
        for index, value in enumerate(my_keys):
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
        # for r in range(rows):
        #    print(dist[r])

        return dist[row][col]

    def string_distance(self, str1, str2):
        if str1.lower() == str2.lower():
            return 0
        l = max(len(str1), len(str2))
        return self.iterative_levenshtein(str1.lower(), str2.lower()) / l;
