from pipeline import *
from fairness import *
from privacy import *

df = load_data("data/adult.csv")
X_train, X_val, X_test, y_train, y_val, y_test = split_data_60_20_20(df)


print(f"Best age threshold: {AGE_THRESHOLD}")
print("\n")




"""
Question 1
"""
print("\n-----------------------------------------\nQuestion 1:")
print("Building classifier and predicting on the test set...")
classifier = train_model(X_train, y_train)
y_pred = predict(classifier, X_test)
print("Done.\n")
print("Evaluation of the classifier on the test set:")
evaluate(y_test, y_pred, to_print=True)


"""
Question 3
"""
print("\n-----------------------------------------\nQuestion 3:")
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


"""
Question 4
"""
print("\n-----------------------------------------\nQuestion 4:")
print("Computing the crosstab of the original dataset...")
original_crosstab = compute_crosstab(df)
print(original_crosstab)

"""
for eps in [0.1, 0.5, 1.0, 2.0]:
    # We do not specify a rng so that we are not biased by an unlucky random seed. (but it will make the results not reproducible so the report will differ)
    crosstabs = []
    errors = []
    estimations_errors = []
    for n in range(10): # We do 10 runs to reduce the variance of the results.
        private_dataset = create_private_dataset(df, eps=eps)
        private_crosstab = compute_crosstab(private_dataset)
        crosstabs.append(private_crosstab)
        error = abs(original_crosstab - private_crosstab)
        errors.append(error)
        estimated_crosstab = estimate_original_crosstab(private_dataset, eps=eps) # Get the estimated value of the crosstab
        estimations_errors.append(abs(original_crosstab - estimated_crosstab))
    print(f"\nEPS {eps} DONE:")
    print(f"\nAverage private crosstabs for eps {eps}:")
    print(sum(crosstabs)/len(crosstabs))
    print(f"\nAverage error for eps {eps}:")
    print(sum(errors)/len(errors))
    print(f"\nAverage estimation error for eps {eps}:")
    print(sum(estimations_errors)/len(estimations_errors))

(See the effect of epsilon in the readme)
"""

#Arbitrary choice of epsilon for the rest of the project
eps = 1.0
rng = np.random.default_rng(CUTTING_SEED) # We set the rng to a fixed seed so that theses results are reproducible.




print("Computing the crosstab of the private dataset...")
private_dataset = create_private_dataset(df, eps=eps, rng=rng)
print(compute_crosstab(private_dataset))
X_train_private, X_val_private, X_test_private, y_train_private, y_val_private, y_test_private = split_data_60_20_20(private_dataset)
print("Training a private classifier on the private dataset and predicting on the private test set...")
private_classifier = train_model(X_train_private, y_train_private)
y_pred_private = predict(private_classifier, X_test_private)
print("Done.\n")
print("Evaluation of the private classifier on the private test set...")
evaluate(y_test_private, y_pred_private, to_print=True)
print("Done.\n")
print("Evaluation of the private classifier on the original test set...")
y_pred = predict(private_classifier, X_test)
evaluate(y_test, y_pred, to_print=True)


"""
Question 5
"""


print("\n-----------------------------------------\nQuestion 5:")
print("Building private+fair classifier and predicting on the private test set...")
Xt_test_private, private_fair_classifier = fair_train_transform(X_train_private, y_train_private, X_test_private)
y_pred_private_fair = fair_predict(private_fair_classifier, Xt_test_private)
print("Done.\n")
print("Evaluation of the fairness metric on the private+fair classifier (using the original test set):")
evaluate_fairness(y_test, y_pred_private_fair, X_test[SENSITIVE], to_print=True)
print("Evaluation of the fairness metric on the fair classifier (using the original test set):")
evaluate_fairness(y_test, y_pred_fair, X_test[SENSITIVE], to_print=True)