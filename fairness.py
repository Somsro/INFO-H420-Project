from fairlearn.metrics import MetricFrame, true_positive_rate
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
from sklearn.linear_model import LogisticRegression
from pipeline import MAX_ITER, CUTTING_SEED, build_preprocessor



SENSITIVE = ["sex", "age_bin"]


def fair_train_transform(X_train, y_train, X_test, max_iter=MAX_ITER):
    """
    Train a fair classifier (Equal Opportunity) with Exponentiated Gradient.

    Parameters:
    - X_train: pandas DataFrame, training features
    - y_train: pandas Series, training target variable
    - X_test: pandas DataFrame, test features
    - (opt) max_iter: int, max iterations of the Logistic Regression

    Returns:
    - Xt_test: array-like, transformed test features
    - mitigator: trained fair classifier
    """
    sensitive_features = X_train[SENSITIVE]
    prep = build_preprocessor()
    Xt_train = prep.fit_transform(X_train) #small t is for transformed. It is a matrix of transformed features.
    Xt_train = Xt_train.toarray() #Fairlearn does not accept sparse matrices, so we convert it to a dense matrix.
    Xt_test = prep.transform(X_test) #Transform the validation/test set with the fitted preprocessor.
    Xt_test = Xt_test.toarray() #Fairlearn does not accept sparse matrices, so we convert it to a dense matrix.

    mitigator = ExponentiatedGradient(LogisticRegression(max_iter=max_iter),
                                  constraints=TruePositiveRateParity())
    mitigator.fit(Xt_train, y_train, sensitive_features=sensitive_features) #Train the fair classifier with the transformed features and the sensitive features.
    return Xt_test, mitigator

def fair_predict(mitigator, Xt_test):
    """
    Predict the income for the given dataset using the trained fair classifier.

    Parameters:
    - mitigator: trained fair classifier
    - Xt_test: pandas DataFrame, transformed features for prediction

    Returns:
    - y_pred: array-like, predicted labels
    """
    return mitigator.predict(Xt_test, random_state=CUTTING_SEED)

def evaluate_fairness(y_test, y_pred, sensitive_features, to_print=False):
    """
    Evaluate the fairness of the model using Equal Opportunity metric.

    Parameters:
    - y_test: array-like, true labels
    - y_pred: array-like, predicted labels
    - sensitive_features: DataFrame, sensitive attributes used for the fairness constraint

    Returns:
    - mf: MetricFrame object containing fairness metrics
    """
    mf = MetricFrame(metrics=true_positive_rate,
                    y_true=y_test, y_pred=y_pred,
                    sensitive_features=sensitive_features)
    if to_print:
        print(f"Fairness metrics: {mf.by_group}") # TPR for each of the 4 groupes (2 age_bin * 2 sex)
        print(f'Biggest gap: {mf.difference(method="between_groups")}')   # max - min

    return mf