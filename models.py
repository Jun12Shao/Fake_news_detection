import numpy as npfrom sklearn.model_selection import RandomizedSearchCV, GridSearchCVimport scipyimport scipy.statsfrom sklearn.svm import SVCfrom sklearn.tree import DecisionTreeClassifierfrom sklearn.ensemble import RandomForestClassifierfrom sklearn.neural_network import MLPClassifierfrom sklearn.metrics import accuracy_scorefrom sklearn.exceptions import ConvergenceWarningimport warningsimport joblibwarnings.filterwarnings(action='ignore', category=ConvergenceWarning)def param_search(classifier,search_type, params, X_train, y_train, X_valid, y_valid, name):    n_iter = 5    if search_type=='random':        param_searcher = RandomizedSearchCV(classifier, params, n_iter=n_iter, verbose=1, random_state=0, cv=3, iid=False, n_jobs=16)        print("    ---> performing random search...")    else :        param_searcher = GridSearchCV(classifier, params, verbose=1, cv=3, iid=False, n_jobs=16)        print("    ---> performing grid search...")    param_searcher.fit(X_train, y_train)    print("    ---> best parameters:", param_searcher.best_params_)    print("    ---> %.1f%% accuracy on validation sets (average)" % (param_searcher.best_score_*100))    joblib.dump(param_searcher.best_estimator_, name+'.sav')    accuracy_train = accuracy_score(y_train, param_searcher.best_estimator_.predict(X_train))    accuracy_test  = accuracy_score(y_valid,  param_searcher.best_estimator_.predict(X_valid))    print("    ---> evaluation:")    print("       > %.1f%% train accuracy" % (accuracy_train*100))    print("       > %.1f%% valid accuracy"  % (accuracy_test*100))# =================================================================#                          CLASSIFICATION# =================================================================def svm_cls(X_train, y_train, X_test, y_test, scoring='mse', n_jobs=16):    classifier = SVC(random_state=0)    param_random = {        'C'    : scipy.stats.reciprocal(1.0, 1000.),        'gamma': scipy.stats.reciprocal(0.01, 10.),    }    param_search(classifier, 'random', param_random, X_train, y_train, X_test, y_test,'svm_cls')    # param_grid={    #     'C': np.logspace(0, 3, 4),    #     'gamma':  np.logspace(-2, 1, 4),    # }    # param_search(classifier,'grid', param_grid, X_train, y_train, X_test, y_test,'svm_cls')def dt_cls(X_train, y_train, X_test, y_test, n_jobs=16):    classifier = DecisionTreeClassifier(random_state=0)    param_random = {        'criterion': ['gini', 'entropy'],        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],        'min_samples_split' : [2, 3, 4, 5, 6]    }    param_search(classifier, 'random',param_random, X_train, y_train, X_test, y_test,'decision_tree')    # param_search(classifier, 'grid',param_random, X_train, y_train, X_test, y_test,'decision_tree')def rf_cls(X_train, y_train, X_test, y_test, n_jobs=16):    classifier = RandomForestClassifier(random_state=0, n_jobs=n_jobs)    param_random = {        'criterion'   : ['gini', 'entropy'],        'max_depth'   : [1, 2, 4,  6,  8],        'n_estimators': [1, 4, 8, 32, 128],    }    param_search(classifier,'random', param_random, X_train, y_train, X_test, y_test,'random_forest')    # param_search(classifier,'grid', param_random, X_train, y_train, X_test, y_test,'random_forest')def nn_cls(X_train, y_train, X_test, y_test, n_jobs=16):    classifier = MLPClassifier(random_state=0,max_iter=200)    param_random = {        'hidden_layer_sizes': [(64,), (64, 128, 64)],        'solver'            : ['sgd', 'adam'],        'activation'        : ['logistic','tanh', 'relu'],        'batch_size'        : [64, 128 ],        'learning_rate_init': [0.001, 0.01],    }    param_search(classifier, 'random', param_random, X_train, y_train, X_test, y_test,'nn_cls')    # param_search(classifier, 'grid', param_random, X_train, y_train, X_test, y_test,'nn_cls')define_model = [svm_cls, dt_cls, rf_cls,nn_cls]