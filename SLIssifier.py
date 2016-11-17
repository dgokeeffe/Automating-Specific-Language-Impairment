"""

SLIssifier.py
~~~~~~~~~~~~

This program creates several binary classifiers to distinguish SLI vs. TD
children based on transcripts provided via the CHILDES corpus

Author: David O'Keeffe
"""

from polyssifier import poly, plot
from childes_data2 import CHILDESdata
from sklearn.utils import shuffle
from sklearn.externals import joblib
import shutil
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
np.seterr(divide='ignore', invalid='ignore')

# ============================================================================
# SLIssifier
# ============================================================================


def run_classifiers(data, label, name):
    '''
    Uses polyssifier to run the following classifiers:
        Linear SVM
        Logistic Regression
        Naive Bayes
        Nearest Neighbours
        Random Forest
        Soft Voting Enssemble
    Arguments:
        data - The features to classify
        label - The label corresponding to SLI or TD for each item in data
    '''
    # Adjust number of folds given the dataset
    if name == 'Conti4':  # LOOCV
        scores, confusions, \
            predictions, probs = poly(data, label, n_folds=19, verbose=True,
                                      save=True, scale=True,
                                      feature_selection=True, scoring='f1',
                                      concurrency=1,
                                      exclude=['Multilayer Perceptron',
                                               'SVM'])
    else:  # k = 10
        scores, confusions, \
            predictions, probs = poly(data, label, n_folds=10, verbose=True,
                                      save=True, scale=True,
                                      feature_selection=True, scoring='f1',
                                      concurrency=1,
                                      exclude=['Multilayer Perceptron',
                                               'SVM'])

    plot(scores)
    plt.savefig(name)
    plt.clf()
    return scores


def numpify_arrays(data, label):
    '''
    Converts the data and label to numpy arrays
    Arguments:
        data - The features to classify
        label - The label corresponding to SLI or TD for each item in data
    '''
    np_data = np.array(data)
    np_label = np.array(label)
    # Currently data is a nested list of 60 features, 1039 entries long
    # Needs to be the other way around
    data = np.swapaxes(np_data, 0, 1)
    # Now shuffle the lists for the cross validation to work properly
    data, label = shuffle(data, np_label, random_state=0)
    return data, label


def calculate_f1(confusion):
    true_positive = confusion[0][0]
    false_postitive = confusion[1][0]
    false_negative = confusion[0][1]
    precision = true_positive/(true_positive+false_postitive)
    recall = true_positive/(true_positive+false_negative)
    return 2 * (precision + recall)/(precision * recall)  # f1


def baseline(np_data, np_label):
    '''
    Retreive the baseline stats, by calculating the mean of the TD
    group for MLU words, NDW100, and Total Utterances. Then comparing
    whether the samples are 1.25 SD below the mean.
    '''
    # Confusion matrices
    baseline_confusion = [[0, 0], [0, 0]]

    # Get the features from the data
    MLU_words = np_data[27]
    NDW = np_data[11]
    total_utts = np_data[25]

    # Isolate the TD samples
    deletion_array = []
    for i in range(0, len(MLU_words)):
        if np_label[i] == 1:  # Is SLI
            deletion_array.append(i)

    # Create a new array with only TD
    td_mlu = np.delete(MLU_words, deletion_array)
    td_ndw = np.delete(NDW, deletion_array)
    td_tu = np.delete(total_utts, deletion_array)

    # Calculate the cutoff
    mlu_cutoff = np.std(td_mlu) - np.mean(td_mlu)
    ndw_cutoff = np.std(td_ndw) - np.mean(td_ndw)
    total_utts_cuttoff = np.std(td_tu) - np.mean(td_tu)

    # Create the confusion matricies
    for i in range(0, len(MLU_words)):
        sample_mlu = MLU_words[i]
        sample_ndw = NDW[i]
        sample_tu = total_utts[i]
        true_label = np_label[i]
        if (sample_mlu < mlu_cutoff and sample_ndw < ndw_cutoff) or \
            (sample_mlu < mlu_cutoff and sample_tu < total_utts_cuttoff) or \
                (sample_tu < total_utts_cuttoff and sample_ndw < ndw_cutoff):
            sli = 1
            if sli == true_label:
                baseline_confusion[0][0] += 1  # true positive
            else:
                baseline_confusion[0][1] += 1  # false negative
        else:
            sli = 0
            if sli == true_label:
                baseline_confusion[1][1] += 1  # true negative
            else:
                baseline_confusion[1][0] += 1  # false postive
    # Calculate F1 scores
    baseline_f1 = calculate_f1(baseline_confusion)
    return baseline_f1


if __name__ == '__main__':
    corpora = ['Conti4', 'ENNI', 'Gillam', 'EG', 'all3']

    # Get the dataset
    childes_data = CHILDESdata(preload=True)
    X_data = []
    Y_data = []
    feature_X = []

    # Get the data for each of the corpora
    for corpus in corpora:
        X_data.append(childes_data.get_X_data(corpus))
        Y_data.append(childes_data.get_Y_data(corpus))

    # Convert to numpy arrays
    for i in range(0, len(X_data)):
        np_data, np_label = numpify_arrays(X_data[i], Y_data[i])
        X_data[i] = np_data
        Y_data[i] = np_label

    bcsvfile = open('baseline.csv', 'w', newline='')
    baselinewriter = csv.writer(bcsvfile, delimiter=',')

    # Execute the classifiers and do the feature extraction
    for i in range(0, len(X_data)):
        # Set up the persistent data
        fcsvfile = open(corpora[i] + '-feature_scores.csv', 'a', newline='')
        featurewriter = csv.writer(fcsvfile, delimiter=',')
        scsvfile = open(corpora[i] + '_scores.csv', 'w', newline='')
        scorewriter = csv.writer(scsvfile, delimiter=',')

        # Run the classifiers
        save_png = corpora[i]
        base_f1 = baseline(X_data[i], Y_data[i])
        scores = run_classifiers(X_data[i], Y_data[i], save_png)

        # Write results to file
        baselinewriter.writerow([save_png, base_f1])
        scorewriter.writerow([save_png, base_f1])
        scores.to_csv(scsvfile, mode='a')

        # Must be ran with FEATURE SELECTION OFF
        for j in range(1, 11):
            rf_file = os.getcwd()+'/poly_/models/Random Forest_'+str(j)+'.p'
            forest = joblib.load(rf_file)
            estimator = forest.best_estimator_.steps[1][1]
            importances = estimator.feature_importances_
            std = np.std([tree.feature_importances_ for tree in
                         estimator.estimators_], axis=0)
            indices = np.argsort(importances)[::-1]

            # Print the feature ranking to file
            for f in range(X_data[i].shape[1]):
                featurewriter.writerow([f+1, indices[f],
                                        importances[indices[f]]])

            # Plot the feature importances from random forest
            fig = plt.figure()
            fig.set_size_inches(18.5, 10.5)
            plt.title("Feature importances")
            plt.bar(range(X_data[i].shape[1]), importances[indices],
                    color="r", yerr=std[indices], align="center")
            plt.xticks(range(X_data[i].shape[1]), indices)
            plt.xlim([-1, X_data[i].shape[1]])
            name = save_png + str(j)
            plt.savefig(name)
            plt.clf()

        # Delete the classifiers because polyssifier will reuse otherwise
        shutil.rmtree(os.getcwd()+'/poly_', ignore_errors=False, onerror=None)

'''

My attempt at recreating the feature selection in Gabani, doesn't quite work

def isolate_features(X_data):
    Removes some features from the analysis
    # GROUP 1: Language productivity
    group1_only = X_data[0:6]
    group1_excluded = X_data[6:]

    # GROUP 2: Morphosyntactic skills
    group2_only = X_data[6:11]
    group2_excluded = X_data[0:6] + X_data[11:]

    # GROUP 3: Vocabulary Knowledge
    group3_only = []
    group3_only.append(X_data[11])
    group3_excluded = X_data[0:11] + X_data[12:]

    # Group 4: Speech fluency
    group4_only = X_data[12:15]
    group4_excluded = X_data[0:12] + X_data[15:]

    # GROUP 5: LM probabilites
    group5_only = X_data[15:23]
    group5_excluded = X_data[0:15] + X_data[23:]

    # GROUP 6: Standard Scores
    group6_only = X_data[23:31]
    group6_excluded = X_data[0:23] + X_data[31:]

    # GROUP 7: Sentence Complexity
    group7_only = X_data[31:58]
    group7_excluded = X_data[0:31] + X_data[58:]

    # GROUP 8: Error rates
    group8_only = X_data[58:]
    group8_excluded = X_data[0:58]

    # For feature selection
    only_conti4_fs, excluded_conti4_fs = isolate_features(X_data[0])
    only_enni_fs, excluded_enni_fs = isolate_features(X_data[1])
    only_gillam_fs, excluded_gillam_fs = isolate_features(X_data[2])

    corpus_only = [only_conti4_fs, only_enni_fs, only_gillam_fs]
    corpus_excluded = [excluded_conti4_fs, excluded_enni_fs,
                       excluded_gillam_fs]
    only_features = []
    excluded_features = []
    # For the feature extraction
    for j in range(0, len(only_gillam_fs)):
        only_np_data, only_np_label = \
            numpify_arrays(corpus_only[i][j], Y_data[i])
        excluded_np_data, excluded_np_label = \
            numpify_arrays(corpus_excluded[i][j], Y_data[i])
        only_features.append((only_np_data, only_np_label))
        excluded_features.append((excluded_np_data, excluded_np_label))

    feature_X.append((only_features, excluded_features))
    only_groups = [group1_only, group2_only, group3_only, group4_only,
                   group5_only, group6_only, group7_only, group8_only]
    excluded_groups = [group1_excluded, group2_excluded, group3_excluded,
                       group4_excluded, group5_excluded, group6_excluded,
                       group7_excluded, group8_excluded]

    return only_groups, excluded_groups

    # Now run the feature selection
    group_labels = ['Conti4_O1', 'Conti4_02', 'Conti4_03', 'Conti4_04',
                    'Conti4_05', 'Conti4_06', 'Conti4_07', 'Conti4_08',
                    'Conti4_E1', 'Conti4_E2', 'Conti4_E3', 'Conti4_E3',
                    'Conti4_E4', 'Conti4_E5', 'Conti4_E6', 'Conti4_E7',
                    'Conti4_E8', 'ENNI_O1', 'ENNI_O2', 'ENNI_O3',
                    'ENNI_O4', 'ENNI_O5', 'ENNI_O6', 'ENNI_O7',
                    'ENNI_O8', 'ENNI_P1', 'ENNI_P2', 'ENNI_P3',
                    'ENNI_P4', 'ENNI_P5', 'ENNI_P6', 'ENNI_P7',
                    'ENNI_P8', 'Gillam_O1', 'Gillam_O2', 'Gillam_O3',
                    'Gillam_O4', 'Gillam_O5', 'Gillam_O6', 'Gillam_O7',
                    'Gillam_O8', 'Gillam_P1', 'Gillam_P2', 'Gillam_P3',
                    'Gillam_P4', 'Gillam_P5', 'Gillam_P6', 'Gillam_P7',
                    'Gillam_P8']
    gl_index = 0  # Group label index
    for corpus in feature_X:
        for incl_excl in corpus:
            for data, label in incl_excl:
                tag = [group_labels[gl_index], 'hi']
                featurewriter.writerow(tag)
                save_png = group_labels[gl_index] + '.png'
                scores = run_classifiers(data, label, save_png)
                scores.to_csv(fcsvfile, mode='a')
                gl_index += 1
'''
