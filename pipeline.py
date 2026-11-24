import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

AGE_THRESHOLD = 39 #If age >= 39, then age_bin feature will be 1, else 0
#Do not change the value, it was given by seek_best_age_threshold function.

CUTTING_SEED = 43 #Used to seperate dataset into training, validation and test.

MISSING_COLUMNS = ["workclass", "occupation", "native-country"] #Columns with missing values

MAX_ITER = 1000 #Max number of iterations for the Logistic Regression model.
NUM = ["education-num", "capital-gain", "capital-loss", "hours-per-week"] #Numerical features
CAT = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"] #Categorical features
BIN = ["age_bin"] #Binary features

def binarize_age(df, age_threshold=AGE_THRESHOLD):
    """
    Binarize the age feature in the dataset based on the given threshold.
    """
    df = df.copy()
    df["age_bin"] = (df["age"] >= age_threshold).astype(int)
    return df

def load_data(file_path, age_threshold=AGE_THRESHOLD):
    """
    Load the dataset from a CSV file.
    
    Parameters:
    - file_path: str, path to the CSV file
    
    Returns:
    - df: pandas DataFrame, loaded dataset
    """
    df = pd.read_csv(file_path, skipinitialspace=True, na_values="?")

    df["income"] = df["income"].str.strip().eq(">50K").astype(int)  # Convert income to binary (0 if <= 50K, else 1).

    # Fill missing values (only found in workclass, occupation, and native-country) with "Unknown" and drop unnecessary columns
    df[MISSING_COLUMNS] = df[MISSING_COLUMNS].fillna("Unknown")

    df = df.drop(columns=["fnlwgt", "education"]) #Drop because they do not provide useful information on a person.
    df = binarize_age(df, age_threshold)
    return df


def split_data_60_20_20(df):
    """
    Split the dataset into training, validation, and test sets following a 60-20-20 split.

    Returns:
    - X_train: pandas DataFrame, training features
    - X_val: pandas DataFrame, validation features
    - X_test: pandas DataFrame, test features
    - y_train: pandas Series, training target variable
    - y_val: pandas Series, validation target variable
    - y_test: pandas Series, test target variable
    """
    X, y = df.drop(columns=["income"]), df["income"] #X = features, y = target variable (income)

    # 60% training set (X_train, y_train), rest 40% will be split into validation and test sets (X_val, y_val, X_test, y_test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=CUTTING_SEED, stratify=y) #stratify=y to make sure there is same % of income = 0 (or 1) in each set.
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=CUTTING_SEED, stratify=y_tmp)

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_preprocessor(num=NUM, cat=CAT, bin=BIN):
    """
    Build a preprocessor for the dataset using ColumnTransformer.

    Parameters:
    - num: list of str, numerical features
    - cat: list of str, categorical features
    - bin: list of str, binary features

    Returns:
    - prep: ColumnTransformer, preprocessor for the dataset
    """
    prep = ColumnTransformer([("num", StandardScaler(), num), #Use standard scaler so high values (such as capital-gain) do not dominate the model.
                            ("cat", OneHotEncoder(handle_unknown="ignore"), cat), #Convert categorical features to 0 or 1
                            ("bin", "passthrough", bin)]) #age_binary is already 0 or 1, so we pass through.
    
    return prep

def train_model(X_train, y_train, num=NUM, cat=CAT, bin=BIN):
    """
    Train a classifier given the training data.

    Parameters:
    - X_train: pandas DataFrame, training features
    - y_train: pandas Series, training target variable
    - (opt) num: list of str, numerical features
    - (opt) cat: list of str, categorical features
    - (opt) bin: list of str, binary features

    Returns:
    - clf: trained classifier
    """

    prep = build_preprocessor(num, cat, bin)
    clf = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=MAX_ITER))])
    clf.fit(X_train, y_train)
    return clf

def predict(clf, X):
    """
    Predict the income for the given dataset using the trained classifier.

    Parameters:
    - clf: trained classifier
    - X: pandas DataFrame, features for prediction

    Returns:
    - y_pred: array-like, predicted labels
    """
    return clf.predict(X)



def evaluate(y_true, y_pred, to_print=False):
    """
    Evaluate the model on the given dataset.

    Parameters:
    - y_pred: array-like, predicted labels
    - y_true: array-like, true labels
    - to_print: bool, print the evaluation metrics if True

    Returns:
    - accuracy: float, accuracy score
    - f1: float, F1 score
    - report: str, classification report
    """
    accuracy = accuracy_score(y_true, y_pred) #y_test is the true income
    f1 = f1_score(y_true, y_pred)
    
    if to_print:
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")

    return accuracy, f1

def seek_best_age_threshold(file_path):
    """
    Seek the best age threshold for binarizing the age feature in the dataset.
    
    Parameters:
    - file_path: str, path to the CSV file
    
    Returns:
    - best_threshold: int, the age threshold that yields the highest F1 score
    """
    df = load_data(file_path)
    best_f1 = 0
    best_threshold = AGE_THRESHOLD

    for threshold in range(20, 75):  # Test thresholds from 20 to 70
        df_binarized = binarize_age(df, age_threshold=threshold)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_60_20_20(df_binarized)
        clf = train_model(X_train, y_train, NUM, CAT, BIN)
        y_pred = predict(clf, X_val)
        _, f1, _ = evaluate(y_val, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


"""
print(seek_best_age_threshold("data/adult.csv")) #Find the best age threshold for binarizing the age feature in the dataset.
(it returns 39)
"""