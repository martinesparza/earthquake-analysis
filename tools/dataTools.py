import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyaldata as pyal
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from typing import Callable
from tools.viz import utilityTools as utility
from tools import params

def get_data_array(data_list: list[pd.DataFrame], trial_cat = 'values_Sol_direction', epoch: Callable = None, area: str ='M1', model: Callable = "pca", n_components:int = 10 ) -> np.ndarray:
    """
    Applies the `model` to the `data_list` and return a data matrix of the shape: sessions x targets x trials x time x modes
    with the minimum number of trials and timepoints shared across all the datasets/targets.
    
    Parameters
    ----------
    `data_list`: list of pd.dataFrame datasets from pyalData (could also be a single dataset)
    `epoch`: an epoch function of the type `pyal.generate_epoch_fun()`
    `area`: area, either: 'PFC', or 'PFC_removed_var', ...
    `model`: a model that implements `.fit()`, `.transform()` and `n_components`. By default: `PCA(10)`. If it's an integer: `PCA(integer)`.
    `n_components`: use `model`, this is for backward compatibility
    'trial_cat': str representing category by which trials are grouped: eg 'cue_id','target_id'
    Returns
    -------
    `AllData`: np.ndarray

    Signature
    -------
    AllData = get_data_array(data_list, delay_epoch, area='PFC', model=10)
    all_data = np.reshape(AllData, (-1,10))
    """
    if isinstance(data_list, pd.DataFrame):
        data_list = [data_list]
    if model is None:
        model = PCA(n_components=n_components, svd_solver='full')
        field_name = "_pca"
    elif isinstance(model, int):
        model = PCA(n_components=model, svd_solver='full')
        field_name = "_pca"
    elif model == 'pca':
        model = PCA(n_components=n_components,svd_solver='full')
        field_name = "_pca"
    else:
        raise ValueError("Invalid model specified. Choose 'isomap', 'pca', or specify number of components for PCA.")
    
    field = f'{area}_rates'
    
    n_shared_trial = np.inf
    target_ids = np.unique(data_list[0][trial_cat])

    for df in data_list:
        for target in target_ids:
            df_ = pyal.select_trials(df, df[trial_cat]== target)
            n_shared_trial = np.min((df_.shape[0], n_shared_trial))
    n_shared_trial = int(n_shared_trial)

    # finding the number of timepoints
    if epoch is not None:
        df_ = pyal.restrict_to_interval(data_list[0],epoch_fun=epoch)
    n_timepoints = int(df_[field][0].shape[0])

    # pre-allocating the data matrix
    AllData = np.empty((len(data_list), len(target_ids), n_shared_trial, n_timepoints, model.n_components))

    rng = np.random.default_rng(12345)
    for session, df in enumerate(data_list):
        df_ = pyal.restrict_to_interval(df, epoch_fun=epoch) if epoch is not None else df
        rates = np.concatenate(df_[field].values, axis=0)
        rates_model = model.fit(rates)
        df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, field_name)
        for targetIdx,target in enumerate(target_ids):
            df__ = pyal.select_trials(df_, df_[trial_cat]==target)
            all_id = df__.trial_id.to_numpy()
            # to guarantee shuffled ids
            while ((all_id_sh := rng.permutation(all_id)) == all_id).all():
                continue
            all_id = all_id_sh
            df__ = pyal.select_trials(df__, lambda trial: trial.trial_id in all_id[:n_shared_trial])
            for trial, trial_rates in enumerate(df__._pca):
                AllData[session,targetIdx,trial, :, :] = trial_rates
 
    return AllData

def plot_VAF(ax, data_list: list[pd.DataFrame], epoch: Callable=None, areas=["all"], model: Callable=None, n_components=10, n_neighbors=10) -> np.ndarray:
    '''
    Plot VAF for each area in areas list, averaged across sessions in data_list, with shaded errorbars.
    '''
    if isinstance(data_list, pd.DataFrame):
        data_list = [data_list]
    if isinstance(areas, str):
        areas = [areas]
    if model is None:
        model = PCA(n_components=n_components, svd_solver='full')
        field_name = "_pca"
    elif isinstance(model, int):
        model = PCA(n_components=model, svd_solver='full')
        field_name = "_pca"
    elif model == 'pca':
        model = PCA(n_components=n_components, svd_solver='full')
        field_name = "_pca"
    elif model == 'isomap':
        model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        field_name = "_isomap"
    
    x_ = np.arange(1, n_components + 1)
    for area in areas:
        field = f'{area}_rates'
        VAF_per_area = []
        for session, df in enumerate(data_list):
            df_ = pyal.restrict_to_interval(df, epoch_fun=epoch) if epoch is not None else df
            rates = np.concatenate(df_[field].values, axis=0)
            rates_model = model.fit(rates)
            if isinstance(model, PCA):
                explained_variance_ratio = model.explained_variance_ratio_
            elif isinstance(model, Isomap):
                raise KeyError('VAF not implemented for Isomap')
            cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
            VAF_per_area.append(cumulative_explained_variance_ratio)
        utility.shaded_errorbar(ax, x_,  np.array(VAF_per_area).T ,label = area,color = getattr(params.colors, area, "k"))
    ax.set_xlabel('Number of PCs ')
    ax.set_ylabel('VAF (%)')
    ax.set_title('Variance accounted for by PCs')
    ax.axhline(y=0.8, color='red', linestyle='--')
    ax.legend()
    plt.show()

def plot_pairwise_corr(ax,df,areas,epoch):
    '''
    Plot pairwise correlation for one session for each area in areas list.
    '''
    if isinstance(areas, str):
        areas = [areas]
    if len(areas) == 1:
        ax = [ax]
    for i, area in enumerate(areas):
        field = f'{area}_rates'  
    
        df_ = pyal.restrict_to_interval(df, epoch_fun=epoch) if epoch is not None else df
        rates = np.concatenate(df_[field].values, axis=0)  
        correlation_matrix = np.corrcoef(rates.T)  
        sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap=params.colors.corr_cmap, cbar=True, ax=ax[i])
        
        # Set titles and labels
        ax[i].set_title(area)
        ax[i].set_xlabel("Neuron #")
        ax[i].set_ylabel("Neuron #")
        plt.show()


 