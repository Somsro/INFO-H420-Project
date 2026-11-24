# INFO-H420-Project
Data found at : https://archive.ics.uci.edu/dataset/2/adult

The age threshold was found by running the function seek_best_age_threshold() in pipeline.py. It returns 39, which will be used during all the project. (it was found by testing thresholds from 20 to 70 and selecting the one that gave the highest F1 score on the validation set)

Fairness metric chosen : Equal opportunity. Was done so the model aims to have equal true positive rates for both (age, sex) groups. Demographic parity was not chosen because it would hide the inequalities in the society between men and women. 
Chosing equal opportunity might reduce the accuracy and f1 scores of the model since there is not a lot of data for the minority groups (female and age >= 39) so the TPR can be unstable. However, on the contrary of demographic parity, choosing equal opportunity does not affect the accuracy of the model concerning predictions for an income of less than 50K. 
The fair classifier was trained using the ExponentiatedGradient algorithm. Then, the evaluation of the classifier and the fair classifier was done using the MetricFrame class from the fairlearn library. 

The results are shown below. We can see a notable difference (66%) in the fairness metric (Equal opportunity) between the classifier and the fair classifier. The accuracy descreased by 2% and the F1 score decreased by 9%. This shows the trade-off between fairness and accuracy.


## Equal opportunity : TPR par groupe

| Group | Classifier | Fair classifier | Difference |
|---|---|---|---|
| Woman, Young (`age_bin=0`) | 0,442 | 0,477 | +0,035 |
| Woman, Old (`age_bin=1`) | 0,470 | 0,503 | +0,034 |
| Man, Young (`age_bin=0`) | 0,501 | 0,545 | +0,044 |
| Man, Old (`age_bin=1`) | 0,640 | 0,496 | −0,144 |
| **Biggest gap** | **0,198** | **0,068** | **−0,130** |

## Fairness metrics

| Metric | Classifier | Fair classifier | Difference |
|---|---|---|---|
| Accuracy | 0,843 | 0,825 | −0,018 |
| F1 score | 0,639 | 0,583 | −0,056 |