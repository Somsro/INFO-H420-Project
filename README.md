# INFO-H420-Project
Data found at : https://archive.ics.uci.edu/dataset/2/adult

the age threshold was found by running the function seek_best_age_threshold() in pipeline.py. It returns 39, which will be used during all the project. (it was found by testing thresholds from 20 to 70 and selecting the one that gave the highest F1 score on the validation set)