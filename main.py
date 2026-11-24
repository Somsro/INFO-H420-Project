from pipeline import *
from fairness import *

df = load_data("data/adult.csv")
X_train, X_val, X_test, y_train, y_val, y_test = split_data_60_20_20(df)


print(f"Best age threshold: {AGE_THRESHOLD}")
print("\n")




"""
Question 1
"""
print("Question 1:\n-----------------------------------------")
print("Building classifier and predicting on the test set...")
classifier = train_model(X_train, y_train)
y_pred = predict(classifier, X_test)
print("Done.\n")
print("Evaluation of the classifier on the test set:")
evaluate(y_test, y_pred, to_print=True)


"""
Question 3
"""
print("Question 3:\n-----------------------------------------")
print("Evaluation of the fairness metric on the classifier :")
evaluate_fairness(y_test, y_pred, X_test[SENSITIVE], to_print=True)

print("Building fair classifier and predicting on the test set...")
Xt_test, fair_classifier = fair_train_transform(X_train, y_train, X_test)
y_pred_fair = fair_predict(fair_classifier, Xt_test)
print("Done.\n")
print("Evaluation of the fairness metric on the fair classifier :")
evaluate_fairness(y_test, y_pred_fair, X_test[SENSITIVE], to_print=True)
print("Evaluation of the fair classifier on the test set:")
evaluate(y_test, y_pred_fair, to_print=True)
