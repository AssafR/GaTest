import random


class MyClassifier:
    def fit(self, x, y):
        pass

    def predict(self, x_test):
        weights = {'hotel_distance': -1}
        bias = 1
        return x_test.apply(lambda row: self.naive_grading_function_2(row, row, weights, bias), axis=1)

    def random_grading_function(self, row1, row2):
        return random.randint(-1, 2)

    def naive_grading_function(self, row1, row2):
        if row1['hotel_distance'] == 0:
            return 1
        return 0

    def naive_grading_function(self, row1, row2):
        """
        Bigger is better
        :param row1:
        :param row2:
        :return:
        """
        val = 1 - float(row1['hotel_distance'])
        if val > 0.8:
            return val
        return 0

    def naive_grading_function_2(self, row1, row2, weights, bias):
        """
        Bigger is better
        :param row1:
        :param row2:
        :return:
        """
        sums = [row1[weight_field] * weights[weight_field] for weight_field in weights.keys()]
        result = bias + sum(sums)
        return result
