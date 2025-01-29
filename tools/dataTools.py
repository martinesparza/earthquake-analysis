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

 