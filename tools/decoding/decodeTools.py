import numpy as np
from tools import dataTools as dt
import tqdm
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tools.params import Params

def within_decoding(cat, allDFs, epoch, area = "M1", model = 10,n_components=10,from_bhv = False, bhv_fields = ["all"], reduce_dim = False, control = False, transformation = None, metric = None, classifier_model =GaussianNB, ax = None):
    '''
    '''
   
    within_score = {}
    target_ids = np.unique(allDFs[0][cat])
    conf_matrices = []
    for i, df in enumerate(allDFs):
        if from_bhv:
            #  for predicting from behavioural data
            AllData = dt.get_data_array_bhv([df], cat, epoch = epoch, bhv_fields=bhv_fields, model = model, n_components=n_components, reduce_dim = reduce_dim, transformation = transformation, metric = metric)
                # _, n_trial, n_comp = AllData1.shape
        else:
            AllData = dt.get_data_array([df], cat, epoch = epoch, area=area,model = model, n_components=n_components)
            AllData = AllData[0,...]
            n_targets,n_trial,n_time,n_comp = AllData.shape
            # print(AllData.shape)
            # resizing
            X = AllData.reshape((-1,n_comp*n_time))
            AllTar = np.repeat(target_ids,n_trial)
            AllTar = np.array(AllTar, dtype=int).flatten()
            # print(AllTar)
            if control:
                np.random.shuffle(AllTar) 
            if ax is not None:
                # Predictions for confusion matrix
                # print(X.shape)
                # print(AllTar.shape)

                y_pred = cross_val_predict(classifier_model(), X, AllTar, cv=10)
                
                # Compute confusion matrix for session
                conf_mat = confusion_matrix(AllTar, y_pred, labels=target_ids)
                conf_matrices.append(conf_mat)

                # Compute accuracy
                within_score[df.session[0]] = np.mean(y_pred == AllTar)
            else:
                _score = cross_val_score(classifier_model(),X,AllTar,scoring='accuracy', cv=10).mean()
                within_score[df.session[0]] = np.mean(_score)

    if ax is not None:            
        avg_conf_matrix = np.mean(conf_matrices, axis=0)
        avg_conf_matrix = avg_conf_matrix.astype('float') / avg_conf_matrix.sum(axis=1)[:, np.newaxis]
        sns.heatmap(avg_conf_matrix, annot=False, fmt='.2f', cmap="Blues", 
                    xticklabels=target_ids, yticklabels=target_ids, ax=ax)
        ax.set_title(f"Predicting {cat} from {area}, score = {np.mean(list(within_score.values())):.2f}, chance = {1/len(target_ids):.2f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    
    return within_score


