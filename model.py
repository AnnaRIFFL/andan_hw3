from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from pickle import dump, load
import pandas as pd

TARGET = '>50K,<=50K'
FEATURES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week']


def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET] == '>50K'

    return X, y


def open_data(path="data.adult.csv"):
    return pd.read_csv(path)


def preprocess_data(df_raw: pd.DataFrame, test=True):
    df = df_raw[~(df_raw == '?').any(axis=1)]
    # df.columns = df.columns.str.replace('-', '_')

    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = df

    for col in X_df.select_dtypes(include=['object']).columns:
        dummy = pd.get_dummies(X_df[col], prefix=col)
        X_df = pd.concat([X_df, dummy], axis=1)
        X_df.drop(col, axis=1, inplace=True)

    # to_encode =
    # to_count = X_df.select_dtypes(include=['number']).columns
    # X_object = pd.get_dummies(X_df[to_encode], drop_first=True)
    # X_count = X_df[to_count]
    # X_df = pd.concat([X_object, X_count], axis=1)

    if test:
        return X_df, y_df
    return X_df


def fit_and_save_model(X_df, y_df, path="data.adult.mw"):
    model = RandomForestClassifier()
    model.fit(X_df, y_df)

    # test_prediction = model.predict(X_df)
    # accuracy = accuracy_score(test_prediction, y_df)
    # print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(X, path="data.adult.mw"):
    try:
        with open(path, "rb") as file:
            model = load(file)
    except FileNotFoundError:
        raise AssertionError("Model file not found. Please train the model first.")

    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X)[0]

    return prediction, prediction_proba


def beautify_results(proba, predict):
    return f"Вы будете получать >50K с вероятностью {proba[1]*100:.1f}%.\n" +\
    f"Те, скорее всего {'' if predict else 'не'} будете"

if __name__ == "__main__":
    df = pd.read_csv('sample_inp.csv')
    load_model_and_predict(df)
