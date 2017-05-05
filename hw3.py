import pandas as pd
import pylab as pl
import numpy as np
import re
import sys
import time
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

# if you're running this in a jupyter notebook, print out the graphs
NOTEBOOK = 0

##importing and formatting your data
def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def print_null_freq(df):
    """
    for a given DataFrame, calculates how many values for each variable is null
    and prints the resulting table to stdout
    """
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    print(pd.crosstab(df_lng.variable, null_variables))

def get_dummies(df, column_name):
	df_result = pd.get_dummies(df[column_name])
	return df_result

def read_data(path):
	df = pd.read_csv(path, index_col=0)
	# column names are in camelCase; easier to read in snake_case
	df.head()

	# convert each column name to snake case
	df.columns = [camel_to_snake(col) for col in df.columns]

	##handling nulls

	# what we really want to do is figure out which variables have
	# null values and handle them accordingly. to do this, we're going to
	# 'melt' our data into 'long' format'
	# the end goal is to have a table that tells us for each variable, how many
	# instances are null vs. non-null
	pd.melt(df)

	return df

def pre_processing(input_df):
	# find out which column has null value, and using mean to fill those null
	# values if there are more than 10.
	null_columns = input_df.columns[input_df.isnull().sum() > 10].tolist()
	for col in null_columns:
		replacement_value = input_df[col].mean()
		input_df[col] = input_df[col].fillna(replacement_value)

	result_df = input_df.dropna()
	return result_df

def get_related_features(input_df, target_feature, related_feature_size):
	features = np.array(input_df.columns)
	# remove target_feature from all features
	index = np.argwhere(features==target_feature)
	features = np.delete(features, index)
	##feature selection

	train_x, test_x, train_y, test_y = train_test_split(input_df[features], input_df[target_feature], test_size=0.25)

	clf = RandomForestClassifier()
	clf.fit(train_x, train_y)

	# from the calculated importances, order them from most to least important
	# and make a barplot so we can visualize what is/isn't important
	importances = clf.feature_importances_
	sorted_idx = np.argsort(importances)
	# Return only the top features up to the related_feature_size.
	related_features = features[sorted_idx[-related_feature_size:]]

	padding = np.arange(len(features)) + 0.5
	pl.barh(padding, importances[sorted_idx], align='center')
	pl.yticks(padding, features[sorted_idx])
	pl.xlabel("Relative Importance")
	pl.title("Variable Importance")
	#pl.show()
	return related_features

def explore_feature(input_df, test_feature, target_feature):
	## exploring features
	age_means = input_df[[test_feature, target_feature]].groupby(test_feature).mean()
	age_means.plot()

	input_df["age_bucket"] = pd.cut(input_df.age, range(-1, 110, 10))
	pd.crosstab(input_df.age_bucket, input_df.serious_dlqin2yrs)
	input_df[["age_bucket", "serious_dlqin2yrs"]].groupby("age_bucket").mean()
	input_df[["age_bucket", "serious_dlqin2yrs"]].groupby("age_bucket").mean().plot()

def define_clfs_params(grid_size):

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def f1(y_true, y_scores, k):
	preds_at_k = generate_binary_at_k(y_scores, k)
	return f1_score(y_true, preds_at_k, average='binary')

def accuracy(y_true, y_scores, k):
	preds_at_k = generate_binary_at_k(y_scores, k)
	return accuracy_score(y_true, preds_at_k)

def recall(y_true, y_scores, k):
	preds_at_k = generate_binary_at_k(y_scores, k)
	return recall_score(y_true, preds_at_k, average='binary')

def precision_at_k(y_true, y_scores, k):
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision

def eval_model_precision_recall(y_true, y_prob, model_name):
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    pl.clf()
    fig, ax1 = pl.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    pl.title(name)
    #pl.savefig(name)
    pl.show()
    
def clf_loop(models_to_run, clfs, grid, X, y):
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'time_used', 'auc-roc','p_at_5','recall_at_5','accuracy_at_5','f1_at_5',
        'p_at_10','recall_at_10','accuracy_at_10','f1_at_10', 'p_at_20','recall_at_20','accuracy_at_20','f1_at_20'))
    for n in range(1, 2):
        # create training and valdation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    start_time = time.time()
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p, elapsed_time, 
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       recall(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       accuracy(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       f1(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       recall(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       accuracy(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       f1(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       recall(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       accuracy(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       f1(y_test_sorted,y_pred_probs_sorted,20.0)]
                    if NOTEBOOK == 1:
                        eval_model_precision_recall(y_test,y_pred_probs,clf)
                        #eval_model_roc(y_test,y_pred_probs,clf)
                except IndexError:
                    print("IndexError")
                    continue
    return results_df

def eval_model_roc(ground_truth, predicts, model_name):
	fpr, tpr, thresholds = roc_curve(ground_truth, predicts)
	roc_auc = auc(fpr, tpr)

	# Plot ROC curve
	pl.clf()
	pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
	pl.plot([0, 1], [0, 1], 'k--')
	pl.xlim([0.0, 1.0])
	pl.ylim([0.0, 1.0])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	name = model_name
	pl.title(name)
	pl.legend(loc="lower right")
	#pl.savefig(name)
	pl.show()

def main():
    df = read_data("credit-data.csv")
    #print_null_freq(df)
    df = pre_processing(df)
    #print_null_freq(df)
    features = get_related_features(df, "serious_dlqin2yrs", 5)
    grid_size = 'test'
    clfs, grid = define_clfs_params(grid_size)
    models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB']
    X = df[features]
    y = df.serious_dlqin2yrs
    results_df = clf_loop(models_to_run, clfs,grid, X,y)
    results_df.to_csv('results.csv', index=False)

if __name__ == "__main__":
   main()


