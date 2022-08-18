from pandas import read_excel
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, KFold, train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report, f1_score, mean_squared_log_error, recall_score, accuracy_score

# Basic ML training given a LIST of models
def basic_training_by_model_list(_X_train, _X_test, _y_train, _y_test, _models_list):
    #To save information
    f1_scores, accuracies = [], []
    # Loop over every model
    for model in _models_list:
        # Training
        model.fit(_X_train,_y_train)
        y_pred = model.predict(_X_test)
        # Evaluating
        acc = accuracy_score(_y_test, y_pred)
        f1scr = f1_score(_y_test, y_pred)
        accuracies.append(acc)
        f1_scores.append(f1scr)
        # Showing results
        #print("Model {} -- F1-score {:.3f} , accuracy is {:.3f}".format(model,f1scr,acc))
    #print('\n')
    return accuracies, f1_scores

# Add numbers to a point in a plot
def annotate_points(ax, list_points, move=False, move_nbr=0):
    if move:
        xx = np.arange(1,len(list_points)*2,2)
    else:
        xx = np.arange(0,len(list_points)*2,2)

    # Annotating points
    for i, txt in enumerate(["{0:0.3f}".format(i) for i in list_points]):
        ax.annotate(txt, (xx[i], list_points[i]-move_nbr))
    return ax

# Plot comparing accuracy and f1-score per model and per re-scaling technique
def plot_score_comparision(acc1, f1scr1, acc2, f1scr2, list_models_names, list_labels, _move_nbr):

    fig, ax1, = plt.subplots(1,1,figsize=(7,5))
    ax1.scatter(np.arange(0,len(acc1)*2, 2), acc1, marker = 'D', s=35, color='cadetblue')
    ax1.scatter(np.arange(0, len(f1scr1)*2, 2), f1scr1, marker = 'x', s=55, color='slateblue')
    ax1.scatter(np.arange(1, len(acc2*2), 2), acc2, marker = 'o', s=35, color='palegreen')
    ax1.scatter(np.arange(1, len(f1scr2)*2, 2), f1scr2, marker = '*', s=55, color='darkblue')

    ax1 = annotate_points(ax1, acc1)
    ax1 = annotate_points(ax1, f1scr1)
    ax1 = annotate_points(ax1, acc2, move= True, move_nbr=_move_nbr)
    ax1 = annotate_points(ax1, f1scr2, move = True, move_nbr=_move_nbr)

    ax1.set_ylabel('F1-score/Accuracy', color = 'black', fontsize = 16)
    ax1.set_xlabel('Default models', color = 'black', fontsize = 16)
    ax1.legend(['Accu '+list_labels[0], 'F1-sco '+ list_labels[0], 'Accu '+ list_labels[1], 'F1-sco '+ list_labels[1]])
    major_ticks = np.arange(0.5,len(list_models_names)*2,2)
    minor_ticks = np.arange(-0.5,len(list_models_names)*2,2)
    ax1.set_xticks(major_ticks, labels = list_models_names)
    ax1.set_xticks(minor_ticks, minor = True)
    plt.grid(which='minor', axis='x', alpha = 0.5)
    
    plt.show

# Function to obtain scores from two different metrics
def CV_multiple_metrics(model, _X, _y, _cv,):
    metrics = ['accuracy', 'f1']
    results = cross_validate(estimator = model, X =_X, y =_y, cv = _cv, 
                            scoring = metrics, return_train_score=True)
    return [results['train_accuracy']], [results['train_accuracy'].mean()], [results['train_f1']], [results['train_f1'].mean()]

# Training using cross validation for multimple metrics and a list of models
def train_x_modelList(models_list, _X, _y, _cv):
    acc_cv_x_model = []
    f1_cv_x_model = []
    for model in models_list:
        _, res_acc, _, res_f1 = CV_multiple_metrics(model, _X, _y, _cv)
        acc_cv_x_model.append(res_acc)
        f1_cv_x_model.append(res_f1)
    
    acc_cv_x_model = [item for sublist in acc_cv_x_model for item in sublist]
    f1_cv_x_model = [item for sublist in f1_cv_x_model for item in sublist]

    return acc_cv_x_model, f1_cv_x_model

# This function plots two lists: accuracy and f1-scores. They must be represented as ONE poin per model
def plotting(_models_names, _accuracies, _f1_scores, title):
    fig, ax1, = plt.subplots(1,1,figsize=(7,6))
    xx = np.arange(len(_accuracies))
    ax1.scatter(xx, _accuracies, marker = 'D', s=35, color='cadetblue')
    ax1.scatter(xx, _f1_scores, marker = 'o', s=55, color='slateblue')

    ax1.set_ylabel('F1-score/Accuracy', color = 'black', fontsize = 16)
    ax1.set_ylabel('Default models', color = 'black', fontsize = 16)
    ax1.legend(['Accuracy', 'F1-score'])
    ax1.set_title(title)

    # Annotating points
    for i, txt in enumerate(["{0:0.2f}".format(i) for i in _accuracies]):
        ax1.annotate(txt, (xx[i], _accuracies[i]-0.015))

    # Annotating points
    for i, txt in enumerate(["{0:0.2f}".format(i) for i in _f1_scores]):
        ax1.annotate(txt, (xx[i], _f1_scores[i]+0.015))

    plt.xticks(ticks=np.arange(0,5), labels=_models_names)

    plt.show()

