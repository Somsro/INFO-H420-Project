# INFO-H420-Project
Data found at : https://archive.ics.uci.edu/dataset/2/adult

The age threshold was found by running the function seek_best_age_threshold() in pipeline.py. It returns 39, which will be used during all the project. (it was found by testing thresholds from 20 to 70 and selecting the one that gave the highest F1 score on the validation set)

The classifier was trained using logistic regression. The results are : accuracy = 0.843, F1 score = 0.639.


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

## Privacy metrics
### Cross tab of the original dataset
Using the crosstab() method from pandas, we can see the distribution of the sensitive attributes (sex and age). They are shown bellow.
| sex    | age = 0 | age = 1 | Total  |
|--------|------------:|------------:|-------:|
| Female |       6 306 |       4 465 | 10 771 |
| Male   |      11 202 |      10 588 | 21 790 |
| **Total** |  17 508  |      15 053 | 32 561 |

as seen in the course, we can use the sequential composition which explains the choice of eps/2 for each of the sensitive attributes (sex and age).

the probability of a noise depending on the epsilon value as well as the results of estimation (averaged over 10 runs, explaining the float values) for each epsilon value is shown below.

| Total ε | ε per attribute | Probability of keeping the truth | Probability of flipping |
|---:|---:|---:|---:|
| 0.1 | 0.05 | 51.25% | 48.75% |
| 0.5 | 0.25 | 56.22% | 43.78% |
| 1.0 | 0.50 | 62.25% | 37.75% |
| 2.0 | 1.00 | 73.11% | 26.89% |

### Average private crosstabs compared with the original

| ε | Sex | Original (age_bin 0) | Private average (age_bin 0) | Original (age_bin 1) | Private average (age_bin 1) |
|---:|---|---:|---:|---:|---:|
| 0.1 | Female | 6306 | 8138.6 | 4465 | 8083.7 |
| 0.1 | Male | 11202 | 8177.9 | 10588 | 8160.8 |
| 0.5 | Female | 6306 | 7850.2 | 4465 | 7788.8 |
| 0.5 | Male | 11202 | 8535.2 | 10588 | 8386.8 |
| 1.0 | Female | 6306 | 7622.0 | 4465 | 7318.6 |
| 1.0 | Male | 11202 | 8984.5 | 10588 | 8635.9 |
| 2.0 | Female | 6306 | 7221.7 | 4465 | 6522.9 |
| 2.0 | Male | 11202 | 9601.4 | 10588 | 9215.0 |

### Average difference between the original and private crosstabs

| ε | Sex | age_bin 0 | age_bin 1 |
|---:|---|---:|---:|
| 0.1 | Female | 1832.6 | 3618.7 |
| 0.1 | Male | 3024.1 | 2427.2 |
| 0.5 | Female | 1544.2 | 3323.8 |
| 0.5 | Male | 2666.8 | 2201.2 |
| 1.0 | Female | 1316.0 | 2853.6 |
| 1.0 | Male | 2217.5 | 1952.1 |
| 2.0 | Female | 915.7 | 2057.9 |
| 2.0 | Male | 1600.6 | 1373.0 |

### Average estimation error

| ε | Sex | age_bin 0 | age_bin 1 |
|---:|---|---:|---:|
| 0.1 | Female | 65198.102562 | 65096.410887 |
| 0.1 | Male | 66835.892550 | 66734.200875 |
| 0.5 | Female | 1744.985852 | 2080.638774 |
| 0.5 | Male | 1709.448147 | 1850.133330 |
| 1.0 | Female | 865.313189 | 969.542169 |
| 1.0 | Male | 855.585083 | 959.814063 |
| 2.0 | Female | 179.047270 | 156.442630 |
| 2.0 | Male | 226.153124 | 188.034428 |

### Private classifier results
As arbitrary choice, we will use the epsilon value of 1.0 for the rest of the project.

The private classifier achieveed an accuracy of 84,11% and a F1 score of 0.631. Compared with 84.34% and 0.639 for the original classifier, we can see that the accuracy decreased by 0.23% and the F1 score decreased by 0.008 so the privacy had a very small impact on the performance of the model.