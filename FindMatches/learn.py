from sklearn.linear_model import LogisticRegression


class learnData:
    def __init__(self):
        pass

    def learn(self, x_train, x_test, y_train, y_test ):
        logisticregr = LogisticRegression()
        logisticregr.fit(x_train, y_train)
        predictions = logisticregr.predict_proba(x_test)
        return predictions

