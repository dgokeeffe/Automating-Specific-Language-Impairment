"""

run_classifiers.py
~~~~~~~~~~~~

This program creates several binary classifiers to distinguish SLI vs. TD
children based on transcripts provided via the CHILDES corpus

Author: David O'Keeffe
"""

from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

def loocv_lpocv(data, label):
    clf = make_pipeline(StandardScaler(),
                        SelectFromModel(RandomForestClassifier(
                                        n_estimators=100,
                                        random_state=42),
                        threshold="median"),
                        LogisticRegression(C=1))
    cv = LeaveOneOut()
    scores = cross_val_score(clf, data, label, cv=cv)
    print("LG LOOCV Accuracy: %0.2f (+/- %0.2f)" %
          (scores.mean(), scores.std() * 2))
    clf = make_pipeline(StandardScaler(),
                        SelectFromModel(RandomForestClassifier(
                                        n_estimators=100,
                                        random_state=42),
                        threshold="median"),
                        SVC(gamma=0.01, C=10, kernel="rbf"))
    scores = cross_val_score(clf, data, label, cv=cv)
    print("SVC LOOCV Accuracy: %0.2f (+/- %0.2f)" %
          (scores.mean(), scores.std() * 2))


def best_n_features(data, label):
    # Scale the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc,
                  step=1,
                  cv=StratifiedKFold(2),
                  scoring='accuracy')
    rfecv.fit(data, label)
    print("Optimal number of features : %d" % rfecv.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


def run_classifiers(data, label):
    # Scale the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Split the dataset
    X_sensor_train, X_sensor_test, \
        y_sensor_train, y_sensor_test = train_test_split(data,
                                                         label,
                                                         random_state=1)
    # Set up the feature selected datasets
    # Recursive Feature Elmination
    select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=40)
    select.fit(X_sensor_train, y_sensor_train)
    X_train_rfe = select.transform(X_sensor_train)
    X_test_rfe = select.transform(X_sensor_test)
    # Model based Feature Selection
    select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
    select.fit(X_sensor_train, y_sensor_train)
    X_train_model = select.transform(X_sensor_train)
    X_test_model = select.transform(X_sensor_test)
    # Univariate Model Selection
    select = SelectPercentile(percentile=50)
    select.fit(X_sensor_train, y_sensor_train)
    X_train_uni = select.transform(X_sensor_train)
    X_test_uni = select.transform(X_sensor_test)


    # SVC
    param_grid = [
        {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
        {'C': [0.001, 0.01, 0.1, 1, 10, 100],
         'gamma': [1, 0.1, 0.01, 0.001],
         'kernel': ['rbf']}, ]
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=10)
    # No selection
    grid.fit(X_sensor_train, y_sensor_train)
    print('Best score for SVC: {}'.format(grid.score(X_sensor_test, y_sensor_test)))
    print('Best parameters for SVC: {}'.format(grid.best_params_))
    # RFE
    grid.fit(X_train_rfe, y_sensor_train)
    print('Best score for RFE SVC: {}'.format(grid.score(X_test_rfe, y_sensor_test)))
    print('Best parameters for SVC: {}'.format(grid.best_params_))
    # Model based
    grid.fit(X_train_model, y_sensor_train)
    print('Best score for Model SVC: {}'.format(grid.score(X_test_model, y_sensor_test)))
    print('Best parameters for SVC: {}'.format(grid.best_params_))
    # Univariate
    grid.fit(X_train_uni, y_sensor_train)
    print('Best score for Univariate SVC: {}'.format(grid.score(X_test_uni, y_sensor_test)))
    print('Best parameters for SVC: {}'.format(grid.best_params_))

    # Linear Regression
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=10)
    # No selection
    grid.fit(X_sensor_train, y_sensor_train)
    print('Best score for Logistic Regression: {}'.format(grid.score(X_sensor_test, y_sensor_test)))
    print('Best parameters for Logistic Regression: {}'.format(grid.best_params_))
    # RFE
    grid.fit(X_train_rfe, y_sensor_train)
    print('Best score for RFE Logistic Regression: {}'.format(grid.score(X_test_rfe, y_sensor_test)))
    print('Best parameters for Logistic Regression: {}'.format(grid.best_params_))
    # Model based
    grid.fit(X_train_model, y_sensor_train)
    print('Best score for Model Logistic Regression: {}'.format(grid.score(X_test_model, y_sensor_test)))
    print('Best parameters for Logistic Regression: {}'.format(grid.best_params_))
    # Univariate
    grid.fit(X_train_uni, y_sensor_train)
    print('Best score for Univariate Logistic Regression: {}'.format(grid.score(X_test_uni, y_sensor_test)))
    print('Best parameters for Logistic Regression: {}'.format(grid.best_params_))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200)
    param_grid = {'max_features': ['sqrt', 'log2', 10],
                  'max_depth': [5, 7, 9]}
    grid = GridSearchCV(rf, param_grid, cv=10)
    # No selection
    grid.fit(X_sensor_train, y_sensor_train)
    print('Best score for Random Forest: {}'.format(grid.score(X_sensor_test, y_sensor_test)))
    print('Best parameters for Random Forest: {}'.format(grid.best_params_))
    # RFE
    grid.fit(X_train_rfe, y_sensor_train)
    print('Best score for RFE Random Forest: {}'.format(grid.score(X_test_rfe, y_sensor_test)))
    print('Best parameters for Random Forest: {}'.format(grid.best_params_))
    # Model based
    grid.fit(X_train_model, y_sensor_train)
    print('Best score for Model Random Forest: {}'.format(grid.score(X_test_model, y_sensor_test)))
    print('Best parameters for Random Forest: {}'.format(grid.best_params_))
    # Univariate
    grid.fit(X_train_uni, y_sensor_train)
    print('Best score for Univariate Random Forest: {}'.format(grid.score(X_test_uni, y_sensor_test)))
    print('Best parameters for Random Forest: {}'.format(grid.best_params_))


    # MLP (Nerual Network)
    params = [{'solver': ['sgd'], 'learning_rate': ['constant'], 'momentum': [0],
               'learning_rate_init': [0.2]},
              {'solver': ['sgd'], 'learning_rate': ['constant'], 'momentum': [.9],
               'nesterovs_momentum': [False], 'learning_rate_init': [0.2]},
              {'solver': ['sgd'], 'learning_rate': ['constant'], 'momentum': [.9],
               'nesterovs_momentum': [True], 'learning_rate_init': [0.2]},
              {'solver': ['sgd'], 'learning_rate': ['invscaling'], 'momentum': [0],
               'learning_rate_init': [0.2]},
              {'solver': ['sgd'], 'learning_rate': ['invscaling'], 'momentum': [.9],
               'nesterovs_momentum': [True], 'learning_rate_init': [0.2]},
              {'solver': ['sgd'], 'learning_rate': ['invscaling'], 'momentum': [.9],
               'nesterovs_momentum': [False], 'learning_rate_init': [0.2]},
              {'solver': ['adam'], 'learning_rate_init': [0.01]}]

    grid = GridSearchCV(MLPClassifier(), param_grid=params, cv=3)
    grid.fit(X_sensor_train, y_sensor_train)
    print('Best score for MLP: {}'.format(grid.score(X_sensor_test, y_sensor_test)))
    print('Best parameters for MLP: {}'.format(grid.best_params_))

