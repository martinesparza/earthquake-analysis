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
from tools import params 
import pyaldata as pyal
from tools.viz import utilityTools as utility

def within_decoding(cat, allDFs, epoch, area = "M1", units = None, model = 10,n_components=10,from_bhv = False, bhv_fields = ["all"], reduce_dim = False, control = False, transformation = None, metric = None, classifier_model =GaussianNB, ax = None, trial_conditions = []):
    '''
    '''
   
    within_score = {}
    target_ids = np.unique(allDFs[0][cat])
    conf_matrices = []
    for i, df in enumerate(allDFs):
        for condition in trial_conditions:
            df = pyal.select_trials(df, condition)
        if from_bhv:
            #  for predicting from behavioural data
            AllData = dt.get_data_array_bhv([df], cat, epoch = epoch, bhv_fields=bhv_fields, model = model, n_components=n_components, reduce_dim = reduce_dim, transformation = transformation, metric = metric)
                # _, n_trial, n_comp = AllData1.shape
        else:
            AllData = dt.get_data_array([df], cat, epoch = epoch, area=area, units = units, model = model, n_components=n_components)
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

                y_pred = cross_val_predict(classifier_model(), X, AllTar, cv=5)
                
                # Compute confusion matrix for session
                conf_mat = confusion_matrix(AllTar, y_pred, labels=target_ids)
                conf_matrices.append(conf_mat)

                # Compute accuracy
                within_score[df.session[0]] = np.mean(y_pred == AllTar)
            else:
                _score = cross_val_score(classifier_model(),X,AllTar,scoring='accuracy', cv=5).mean()
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

def plot_decoding_over_time(ax, category, df_list, areas, n_components, model, idx_event = 'idx_sol_on', min_time = 0, max_time = 2, trial_conditions = []) :
    '''
    
    '''
    if isinstance(areas, str):
        areas = [areas]
    if isinstance(areas, dict):
        units_per_area = list(areas.values())
        areas = list(areas.keys())
    else:
        units_per_area = None
    
    within_results_per_area = []
    min_timebin =int(min_time/Params.BIN_SIZE)
    max_timebin = int(max_time/Params.BIN_SIZE)
    for i,area in enumerate(areas):
        within_results_over_time = []
        for timebin in range(min_timebin,max_timebin):
            perturb_epoch = pyal.generate_epoch_fun(start_point_name=idx_event,
                                                rel_start=int(min_timebin),
                                                rel_end=int(timebin)
                                                )
            if units_per_area is not None:
                area = "all"
                units = units_per_area[i]
            else:
                units = None
            within_results = within_decoding(cat = category,  allDFs = df_list, area = area, units= units, n_components = n_components, epoch = perturb_epoch, model = model, trial_conditions=trial_conditions)
            within_results_over_time.append([result for result in within_results.values()])

        within_results_per_area.append(np.array(within_results_over_time))

    
    time_axis = np.arange(min_time,max_time,Params.BIN_SIZE)*1000
    time_axis = time_axis[1:]
    for i, area in enumerate(areas):
        utility.shaded_errorbar(
                ax,
                time_axis,
                within_results_per_area[i],
                label=area,
                color=getattr(params.colors, area, "k"),
            )


    chance_level = 1/len(np.unique(df_list[0][category]))
    ax.set_xlabel("Window length (ms)")
    ax.set_ylabel("Decoding accuracy (%)")
    ax.set_title(f"Decoding accuracy using increasing time intervals")
    ax.axvline(x=0, color="k", linestyle="--", label = idx_event)
    ax.axhline(y=chance_level, color="red", linestyle="--", label = "Chance level")
    ax.legend()