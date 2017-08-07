
# coding: utf-8

# # Machine Learning Project

#!/usr/bin/python
import sys
import pickle
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score,classification_report
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

t=time.time()

# these are all the features
features_list_all = ['poi','salary', 'total_payments', 'bonus', 'deferred_income',
                 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 
                 'from_messages','from_this_person_to_poi', 'shared_receipt_with_poi'] 
                

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#exploratory functions
#to look at the structure of the dictionary:
def find_by_name(name):
    '''
    Enter the name of the person at it appears in the dictionary to retrieve all of her information
    For instance, to find Jeff Skilling, use:
    find_by_name("SKILLING JEFFREY K")
    '''
    for key,value in my_dataset.items():
        if key==name:
            print(key, value)
            break
    return 1

def find_an_interesting_person(fieldname, limit):
    '''
    This function finds all the people whose some "fieldname" value is above a certain "limit"
    For instance to find everyone with a salary above 1000000, use:
    find_an_interesting_person("salary", 1000000)
    '''
    names=[]
    for key,value in my_dataset.items():
        if value[fieldname]!="NaN":
            if int(value[fieldname]) >= limit:
                print(key)
                names.append(key)
    return names

def find_outliers(fieldname):
    '''
    To find and print out statistical outliers for a particular feature using the bounds:
    q1-1.5*iqr and q3+1.5*iqr where q1 and q3 are first and third quartile, and iqr is interquartile range
    If the bounds is above the 100th percentile or below zeroth percentile, then just print out top of bottom
    5% respectively. This function does not remove outliers - it just prints out name of persons and values of that field
    so we can study what to remove and what is a legitimate data point.
    '''
    value_list=[]
    name_list=[]
    outlier_names=[]
    outlier_values=[]
    poi_nan_value=0
    non_poi_nan_value=0
    poi=0
    non_poi=0
    #Arrange the fieldname value into a nice list ignoring NANs
    for key,value in my_dataset.items():
        if value["poi"]==1:
            poi+=1
        else:
            non_poi+=1
        if value[fieldname]!="NaN":
            value_list.append(value[fieldname])
            name_list.append(key)
        else:
            if value["poi"]==1:
                poi_nan_value+=1
            else:
                non_poi_nan_value+=1
    #statistical analysis: calculate IQR and Q1, Q3
    iqr = stats.iqr(value_list)
    q1=np.percentile(value_list, 25)
    q3=np.percentile(value_list, 75)
    low_range=q1-1.5*iqr
    high_range=q3+1.5*iqr
    if low_range < 0:
        low_range = np.percentile(value_list, 5)
    if high_range > max(value_list):
        high_range = np.percentile(value_list, 95)
    #Find all the outlier indices
    outliers=np.where(np.logical_or(value_list<=low_range, value_list>=high_range))[0]
    #print out results
    print("*************"+fieldname+"******************")
    print("total_numbers", len(my_dataset.keys()), poi, non_poi)
    print("Number of NaN: ", poi_nan_value, non_poi_nan_value)
    print(high_range, low_range)
    for i in outliers:
        print(name_list[i]+": ",value_list[i])
        outlier_names.append(name_list[i])
        outlier_values.append(value_list[i])
    return outlier_names,outlier_values


#only getting rid of one outlier which is not a legitimate data point
#there are some other extreme values, but they tend to be POIs and there are so few POIs to train the data on
#that throwing them away seems like a waste
my_dataset.pop("TOTAL", 0)
'''
find_outliers('salary')
find_outliers('total_payments')
find_outliers('bonus')
find_outliers('deferred_income')
find_outliers('total_stock_value')
find_outliers('expenses')
find_outliers('exercised_stock_options')
find_outliers('long_term_incentive')
find_outliers('restricted_stock')
find_outliers('director_fees')
find_outliers('to_messages')
find_outliers('from_poi_to_this_person')
find_outliers('from_messages')
find_outliers('from_this_person_to_poi')
find_outliers('shared_receipt_with_poi')
'''

#this function makes a 3D plot that lets us compare three features
#by making a 3D plot.
def compare_three_features(feature_names, scaled=False, xlimit=0, ylimit=0, zlimit=0, ts_size=0.3):
    '''
    the features name should be in the following format:
    feature_names=['poi', 'from_this_person_to_poi', 'from_poi_to_this_person', 'salary']
    where the first field 'poi' is used to distinguish pois from non-pois
    and the rest are axes of the 3D plot.

    For example, try this:
    feature_names=['poi', 'from_this_person_to_poi', 'from_poi_to_this_person', 'salary']
    compare_three_features(feature_names)

    It allows scaling each feature (by maximum value) to demonstrate if it would be meaning ful to scale this 
    feature while conbining.
    '''
    short_data = featureFormat(my_dataset, feature_names, sort_keys = True)
    short_labels, short_features = targetFeatureSplit(short_data)
    
    #simple train-test split:
    #why split? Because I want to understand the trend, not look at all the information
    #then I might mentally overfit
    x_train, x_test, y_train, y_test = train_test_split(short_features, short_labels, test_size=ts_size, random_state=42)

    #find correlaion between these two:
    first=[]
    second=[]
    third=[]
    for item in x_train:
        first.append(item[0])
        second.append(item[1])
        third.append(item[2])

    if scaled:
        first/=max(first)
        second/=max(second)
        third/=max(third)
    #print("first", first)
    #print("second", second)
    r2=np.corrcoef([first, second, third]) #[0, 1]
    print("correlation coefficient", r2)    
    
    
    #let's divide up the training sample into the two classes:
    #all pois:
    first_poi=[]
    first_non_poi=[]
    second_poi=[]
    second_non_poi=[]
    third_poi=[]
    third_non_poi=[]
    for i in range(0,len(y_train)):
        if y_train[i] == 1:
            first_poi.append(first[i])
            second_poi.append(second[i])
            third_poi.append(third[i])
        else:
            first_non_poi.append(first[i])
            second_non_poi.append(second[i])
            third_non_poi.append(third[i])

    #plot
    fig = plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    
    if xlimit!=0:
        plt.xlim(-10, xlimit)
    if ylimit!=0:
        plt.ylim(-10, ylimit)
    if zlimit!=0:
        plt.zlim(-10, zlimit)
    
    
    ax.scatter(first_non_poi, second_non_poi, third_non_poi, facecolor="blue", s=20, edgecolors="blue", alpha=0.5)
    ax.scatter(first_poi, second_poi, third_poi, facecolor="none", edgecolors="red", s=20)
    ax.set_xlabel(feature_names[1])
    ax.set_ylabel(feature_names[2])
    ax.set_zlabel(feature_names[3])
    plt.show()
    
    return 1


#making a composite feature either be linearly combining scaled features,
#or by combining them radially
def combine_features(feature_names, new_feature, scaled=False, ignore_nan=False, method="linear"):
    '''
    This is an attempt to make a composite feature by either combining them radially = x^2 + y^2
    or linearly = x + y
    Any number of things can be combined this way, and a new dictionary is returned with the dataset added.
    If scaled is True, each field is scaled before combining.
    ignore_nan set to True means even if one field has NaN value, the others are combined, assuming
    the NaN is 0. Not ignoring nan (ignore_nan = False) means we just set the new feature to 0 if any of the 
    component featutes is 0.
    '''
    edited_dict=my_dataset
    
    list_of_features={}
    
    #This part makes list that would help us scale features:
    for name in feature_names:
        list_of_features[name]=[]
        for item in edited_dict.values():
            if item[name]!="NaN":
                list_of_features[name].append(item[name])
        
    #combining features:
    if method=="linear":
        pow_ind=1
    elif method=="radial":
        pow_ind=2
    else:
        raise NameError("Method must be 'linear' or 'radial'")

    #adding new feature:
    for item in edited_dict.values():
        combined_value=0
        nan_flag = 0 
        for name in feature_names:
            if item[name]=="NaN":
                nan_flag = 1
                numer = 0
                item[name] = 0
            else:
                numer = item[name]-min(list_of_features[name])

            if scaled:
                combined_value+=pow(numer/(max(list_of_features[name]) - min(list_of_features[name])), pow_ind)
            else:
                combined_value+=pow(item[name], pow_ind)
            item[new_feature]=combined_value           
        #if one of the fields is nan, and we are not supposed to ignore nan,
        #then set new feature to zero
        if nan_flag==1 and not ignore_nan:
            item[new_feature]=0 

    return edited_dict

#finding the best features:
def find_best_features(dataset=my_dataset, feature_names=features_list_all):
    ### Extract features and labels from dataset for local testing
    short_data = featureFormat(dataset, feature_names, sort_keys = True) #my_dataset or new_dict
    #print("short_data", short_data)
    short_labels, short_features = targetFeatureSplit(short_data)
    short_features=MinMaxScaler().fit_transform(short_features)

    x_train, x_test, y_train, y_test = train_test_split(short_features, short_labels, test_size=0.3, random_state=None, stratify=short_labels)
    print("***********To check automated best features:*****************")
    selector = SelectKBest(chi2, k='all').fit(x_train,y_train)
    #names = selector.get_feature_names()
    scores = selector.scores_
    print(feature_names)
    print(scores)
    return scores

#visualize:
#feature_names = ["poi","bonus", "total_payments", 'shared_receipt_with_poi']
#compare_three_features(feature_names, True, "radial")

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#---------------------------------------------------------------------------------------

#Many rounds of trial and error with many combination of features, finally boiled down to:
#feature_names=["poi", 'from_this_person_to_poi', 'long_term_incentive', "salary"]
'''
print("ATTEMPT 3")
print('Adding salary and bonus linearly:')
feature_names=["salary", "bonus"]
new_dict=combine_features(feature_names, "salary_plus_bonus", True, False, method="linear")
features_list_new = features_list_all
features_list_new.append("salary_plus_bonus")
find_best_features(new_dict, features_list_new)
best_decision_tree_classifier(new_dict, features_list_new)
'''
print("time elapsed to try all the new features:")
print(time.time()-t)
t=time.time()

#simple decision tree for feature selection:
def simple_decision_tree(dataset=my_dataset, feature_names=["poi", "shared_receipt_with_poi",  'from_this_person_to_poi', 'from_poi_to_this_person','long_term_incentive', "salary",'exercised_stock_options', "bonus"]):
    '''
    Just a simple decision tree for feature selection
    '''
    #setting up pipeline estimator lisst:
    clf=tree.DecisionTreeClassifier(min_samples_split=5)
    test_classifier(clf, dataset, feature_names)
    return 1


print("**************************Trying different combinations of features*****************************")
print("***************Initial Intuition:***************")
print("These features:", ['from_this_person_to_poi', 'long_term_incentive', "salary"])
simple_decision_tree(my_dataset, ["poi", 'from_this_person_to_poi', 'long_term_incentive', "salary"])
print("***************Let's try different combinations of the best features using selectKbest***************")
print("These features:", ['bonus', 'long_term_incentive', "salary"])
simple_decision_tree(my_dataset, ["poi", 'bonus', 'long_term_incentive', "salary"])
print("These features:", ['exercised_stock_options', 'long_term_incentive', "salary", "bonus"])
simple_decision_tree(my_dataset, ["poi", 'exercised_stock_options', 'long_term_incentive', "salary", "bonus"])
print("These features:", ['exercised_stock_options', 'total_stock_value',  "salary", "bonus"])
simple_decision_tree(my_dataset, ["poi", 'exercised_stock_options', 'total_stock_value',  "salary", "bonus"])
print("These features:", ['exercised_stock_options', 'total_stock_value',  "salary", "bonus", "shared_receipt_with_poi"])
simple_decision_tree(my_dataset, ["poi", 'exercised_stock_options', 'total_stock_value',  "salary", "bonus", "shared_receipt_with_poi"])

print("SEVEN FEATURES")
simple_decision_tree(my_dataset)

#Maybe ratio of from messages to messages from this person to POI would be an indicator. 
#No do not see any particular trend there if I make a plot.
#-----------------------------------------------------------------------------------------
# ## Automated version

#slighly modified version of test classifier, to be used with 
#gridsearchCV to see what the best parameters for each fold
def grid_search_test_classifier(clf, dataset, feature_list, folds = 5):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    param_dict_list=[]
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        print("Training length", len(features_train), "Test set", len(features_test))
        param_dict_list.append(clf.best_params_)
        print("best params", clf.best_params_)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print("Evaluating performance for processed predictions:")
                break
    best_params=clf.best_params_
    real_dict={}
    for keys, itemss in best_params.items():
        real_dict[keys]=[]
        for each_dict in param_dict_list:
            for key1, item1 in each_dict.items():
                if key1==keys:
                    real_dict[keys].append(item1)
        if isinstance(itemss, int):
            print("*******MEDIAN********* ", keys, np.median(real_dict[keys]))
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print("")
    except:
        print("Got a divide by zero when trying out:", clf)
        print("Precision or recall may be undefined due to a lack of true positive predicitons.")


def automated_fitting(feature_names=["poi", "shared_receipt_with_poi",  'from_this_person_to_poi', 'from_poi_to_this_person','long_term_incentive', "salary",'exercised_stock_options', "bonus"]):
    '''
    This function takes in a list of featues, and runs pca and a list of models along with grid searches 
    for best fit parameters to find the best possible restuls.
    It prints out both the results of cross validation and validation using the tester.py script.
    It uses decisiontree, adaboost, randomforest, extratree, mlp and svc. I removed knearestneighbors.
    It probably does poorly due to "0" values from "NaN".
    '''
    #setting up pipeline estimator lisst:
    print("******************This is the code to tune parameters using grid search CV******************")
    models_to_try = [[('reduce_dim', PCA()), ('clf', tree.DecisionTreeClassifier())],
                   [('reduce_dim', PCA()), ('clf', AdaBoostClassifier())],
                  #[('reduce_dim', PCA()), ('clf', KNeighborsClassifier())],
                     [('reduce_dim', PCA()), ('clf', RandomForestClassifier())]]
                     #[('reduce_dim', PCA()), ('clf', ExtraTreesClassifier())],
                     #[('reduce_dim', PCA()), ('clf', MLPClassifier())],
                    #[('reduce_dim', PCA()), ('clf', SVC())]]

    #setting up parameters
    p_grid=[dict(reduce_dim__n_components=[1, 2, 3, 4, 5, 6], clf__min_samples_split=[2,3,4,5,8,10], clf__presort=("True", "False")),
            dict(reduce_dim__n_components=[1, 2, 3, 4, 5, 6], clf__n_estimators=[2,3,5,7,10,40,50,90,120,160], clf__random_state=[15,54,42,5,0]),#, clf__base_estimator=["None", " DecisionTreeClassifier()"], clf__base_estimator__max_depth=[2, 3, 5, 10, 20, 40]),
            #dict(reduce_dim__n_components=[1, 2, 3, 4, 5, 6], clf__n_neighbors=[5,10,20, 30],clf__weights=["uniform","distance"],clf__algorithm=["auto"]),
            dict(reduce_dim__n_components=[1, 2, 3, 4, 5, 6], clf__n_estimators=[2,3,5,7,10,40,50], clf__min_samples_split=[2,3,4,5,8,10], clf__random_state=[15,54,42,5,0])]
            #dict(reduce_dim__n_components=[1, 2, 3, 4, 5, 6], clf__n_estimators=[2,3,5,7,10,40,50], clf__min_samples_split=[2,3,4,5,8,10], clf__random_state=[15,54,42,5,0]),
            #dict(reduce_dim__n_components=[1, 2, 3, 4, 5, 6],clf__solver=['lbfgs','sgd','adam'], clf__alpha=[1e-5, 1e-4, 1e-3, 1, 10, 100]),
            #dict(reduce_dim__n_components=[1, 2, 3, 4, 5, 6], clf__C=[0.1, 1, 5], clf__gamma=[1e-6, 1e-3, 0.1, 1, 10, 100])]

    model_names=["Decision Tree", "Adaboost", "RandomForest"]#, "ExtraTrees", "MLP", "SVC"] 

    ind=0
    for model in models_to_try:
        pipe=Pipeline(model)
        clf=GridSearchCV(pipe, param_grid=p_grid[ind])
        grid_search_test_classifier(clf, my_dataset, feature_names)
        ind+=1


#automated_fitting()   

### Extract features and labels from dataset for local testing
feature_names=['exercised_stock_options', 'total_stock_value',  "salary", "bonus"]
#["poi", "shared_receipt_with_poi",  'from_this_person_to_poi', 'from_poi_to_this_person','long_term_incentive', "salary",'exercised_stock_options', "bonus"] # 

def best_automated_classifier(algorithm="DecisionTree", dataset=my_dataset, feature_names=["poi", 'exercised_stock_options', 'total_stock_value',  "salary", "bonus"]):
    '''
    Afer trying on several classifiers (decision tree, randomforest, adaboost)
    I decided random forest does the best job
    But the best combination of parameters for the other two can still be commented out and tried.
    Randomforest without any new features shows the best performance.
    But decision tree with "bonus", "total_payments", "shared_receipt_with_poi" scaled and radially added also 
    comes pretty close.
    '''
    #feature scaling only for SVC - decision trees and ensemble methods employing 
    #weak estimators that are dec trees aren't sensitibe to scaling:
    #Best RandomForest:
    if algorithm=="RandomForest":
        pipe_model=[('reduce_dim', PCA(n_components=2)), ('clf', RandomForestClassifier(n_estimators=5, min_samples_split=5, random_state=15, class_weight={1:0.85, 0:0.15}))]
        #0.5 all around
    elif algorithm=="DecisionTree":
        #Best Decision tree:
        pipe_model=[('reduce_dim', PCA(n_components=4)), ('clf', tree.DecisionTreeClassifier(min_samples_split=3, presort="True"))]
        #Accuracy: 0.78462  Precision: 0.33333  Recall: 0.40000 F1: 0.36364 F2: 0.38462
    elif algorithm=="AdaBoost":
        #Best Adaboost:
        pipe_model=[('reduce_dim', PCA(n_components=4)), ('clf', AdaBoostClassifier(n_estimators=7, random_state=15))]
        #Accuracy: 0.84615  Precision: 0.50000  Recall: 0.40000 F1: 0.44444 F2: 0.41667
    else:
        raise NameError("Algorithm must be RandomForest, DecisionTree or AdaBoost")

    #print training and test size:
    print("best automated with "+algorithm)

    #pipe=Pipeline(pipe_model)
    #p_grid=dict(reduce_dim__n_components=[2], clf__n_estimators=[7], clf__random_state=[15])
    #clf=GridSearchCV(pipe, param_grid=p_grid)#, cv=StratifiedKFold(3))
    
    clf=Pipeline(pipe_model)
    
    #print("Best params",clf.best_params_)
    test_classifier(clf, dataset, feature_names)

    return clf, feature_names

print("*************************Creating and Testing Composite Features:*************************")

print("ATTEMPT 1")
print("test by radially adding the following two features:")
feature_names=['shared_receipt_with_poi', 'from_poi_to_this_person']
print(feature_names)
new_dict=combine_features(feature_names, "shared_receipt_from_this_person_to_poi", True, False, method="radial")
features_list_new = features_list_all
features_list_new.append("shared_receipt_from_this_person_to_poi")
find_best_features(new_dict, features_list_new)
best_automated_classifier("AdaBoost", new_dict, features_list_new)


print("ATTEMPT 2")
print('combine "bonus", "total_payments", "shared_receipt_with_poi" radially')
new_dict=combine_features(["bonus", "total_payments", 'shared_receipt_with_poi'], "best_3", True, False, "radial")
features_list_new = features_list_all
features_list_new.append("best_3")
find_best_features(new_dict, features_list_new)
best_automated_classifier("AdaBoost", new_dict, features_list_new)
best_automated_classifier("RandomForest", new_dict, features_list_new)
best_automated_classifier("DecisionTree", new_dict, features_list_new)

print("********************************************************************************************")

print("Final Best automated classifier (not using composite features here) is RandomForest")
print("RandomForest with the seven features I picked is written in .pkl file")
print("But here I print out results for Decision Tree and Adaboost as well.")
print(time.time()-t)
t=time.time()

clf, features_list = best_automated_classifier()
best_automated_classifier("RandomForest")
best_automated_classifier("AdaBoost")

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

'''
#Future suggestions to explore:
#GridSearchCV along with SelectKBest could be used to find the best features. If this of interest, take a look at this template:

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(n_splits = 1000, test_size = 0.30, train_size = 0.70, random_state = 42)
kbest = SelectKBest()

values = # TODO: create a list containing values to be tuned for 'k'.  For eg. [1, 2, 3]
clf = # TODO: specify a classifier to be used in the pipeline below

param_grid = {'kbest__k': values}
pipe = Pipeline([('kbest', kbest), ('name_classifier', clf)])

def select_k_best(pipe, cv, param_grid):
    """Perform feature selection process using GridSearchCV and SelectKBest

    Args: 
        pipe: pipeline object

        cv: int, cross-validation generator or an iterable, optional

        param_grid : dict or list of dictionaries
            Dictionary with parameters names (string) as keys and lists of parameter 
            settings to try as values, or a list of such dictionaries, in which case the 
            grids spanned by each dictionary in the list are explored. This enables searching 
            over any sequence of parameter settings.

    Returns:
        best_clf: estimator object

        best_features: tuple
            Contains a list of the best features
    """
    grid_search = GridSearchCV(estimator = pipe, 
                               param_grid = param_grid,
                               cv = cv,
                               verbose = 1)
    grid_search.fit(features, labels)
    best_clf = grid_search.best_estimator_
    best_features = [features_list[i+1] for i in best_clf.named_steps['kbest'].get_support(indices=True)]
    print('Best Features:', best_features)
    return best_clf, best_features

best_clf, best_features = select_k_best(pipe, cv, param_grid)
'''

