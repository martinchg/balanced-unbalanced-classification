import pandas as pd
from lib import train_test_split_stratified

def test_train_test_split_stratified():
    X = pd.DataFrame({"x1": [1, 2, 3, 4, 5, 6],
                      "x2": [10, 20, 30, 40, 50, 60]})
    y = pd.Series([0, 0, 0, 1, 1, 1])

    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.5, random_state=0)

    # même nb de lignes
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)

    # stratification grossièrement respectée
    assert y_train.mean() == y.mean()