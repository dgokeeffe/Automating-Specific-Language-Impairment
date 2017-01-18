"""

SLIssifier.py
~~~~~~~~~~~~

This program creates several binary classifiers to distinguish SLI vs. TD
children based on transcripts provided via the CHILDES corpus

Author: David O'Keeffe
"""

from childes_data2 import CHILDESdata
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
np.seterr(divide='ignore', invalid='ignore')

# ============================================================================
# SLIssifier
# ============================================================================


def numpify_arrays(data, label, corpus):
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
    data, label = shuffle(data, np_label, random_state=0)
    # Save this data for R
    np.savetxt(("%s_x.csv" % corpus), data, delimiter=",")
    np.savetxt(("%s_y.csv" % corpus), label, delimiter=",")
    return data, label

def extract_features():
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
        np_data, np_label = numpify_arrays(X_data[i], Y_data[i], corpora[i])
        X_data[i] = np_data
        Y_data[i] = np_label


    bcsvfile = open('baseline.csv', 'w', newline='')
    baselinewriter = csv.writer(bcsvfile, delimiter=',')

    # Execute the classifiers and do the feature extraction
    for i in range(0, len(X_data)):
        # Run the classifiers
        scores = loocv_lpocv(X_data[i], Y_data[i])

        # Set up the persistent data
        fcsvfile = open(corpora[i] + '-feature_scores.csv', 'a', newline='')
        featurewriter = csv.writer(fcsvfile, delimiter=',')
        scsvfile = open(corpora[i] + '_scores.csv', 'w', newline='')
        scorewriter = csv.writer(scsvfile, delimiter=',')

        # Write results to file
        baselinewriter.writerow([save_png, base_f1])
        scorewriter.writerow([save_png, base_f1])
        scores.to_csv(scsvfile, mode='a')
