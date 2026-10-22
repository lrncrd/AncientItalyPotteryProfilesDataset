
from scipy import stats
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import  chi2_contingency
import pyefd
from aippa.descriptors import ContourDescriptor
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import directed_hausdorff
from aippa.utils import minimum_image
from skimage import transform
from sklearn.metrics.cluster import mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, v_measure_score, homogeneity_score, completeness_score, fowlkes_mallows_score


############ Metrics ##################

def first_quant (x):
    return np.quantile(x,  q = 0.25)
def third_quant (x):
    return np.quantile(x, q = 0.75) 
def coeff_var (x):
    return stats.variation(x)
def standard_error (x):
    return stats.sem(x)
def coefficient_of_quartile_deviation(x):
    return (third_quant(x) - first_quant(x)) / (third_quant(x) + first_quant(x))






def descriptive_by_groups(info_selected, groups,metrics = ['mean', 'std', standard_error, 'median', 'min', 'max', first_quant, third_quant, coeff_var, coefficient_of_quartile_deviation], fields = ['height (cm)', 'width (cm)', 'volume (l)', 'diameter (cm)', 'surface (mq)']):
    """
    Calculate descriptive statistics for multiple fields grouped by specified groups.

    Parameters:
        info_selected (pandas.DataFrame): The DataFrame containing the selected information.
        groups (str or list): The column(s) in the DataFrame used to group the data.
        metrics (list, optional): The list of metrics to calculate for each field. Defaults to ['mean', 'std', 'standard_error', 'median', 'min', 'max', 'first_quant', 'third_quant', 'coeff_var', 'coefficient_of_quartile_deviation'].
        fields (list, optional): The list of fields in the DataFrame for which to calculate descriptive statistics. Defaults to ['height (cm)', 'width (cm)', 'volume (l)', 'diameter (cm)', 'surface (mq)'].

    Returns:
        pandas.DataFrame: A DataFrame containing the calculated descriptive statistics for each field grouped by the specified groups.

    """
    groups = info_selected.groupby(groups)

    df_size = groups.agg({
        'ids': ['count'],
        **{field: metrics for field in fields}
    })

    return df_size.round(3)



def normality_tests(x, view_scatter = False):
    """
    Performs normality tests on a given dataset x and returns the results in a pandas DataFrame.
    
    Parameters:
    - x (numpy.ndarray): The dataset for which normality tests are to be performed.
    - view_scatter (bool, optional): Flag to display a scatter plot of the data points (default: False).
    
    Returns:
    pandas.DataFrame
    """

    methods = OrderedDict()
    methods["D'Agostino & Pearson test"] = stats.normaltest(x)
    methods["Shapiro-Wilk test"] = stats.shapiro(x)
    methods["Kolmogorov-Sminrnov test"] = stats.kstest(x, "norm")

    res = []

    for i, (label, method) in enumerate(methods.items()):

        res.append([label, method[0], method[1]])

    if view_scatter ==True:

        stats.probplot(x, plot = plt)

    return pd.DataFrame(res, columns = ["Test", "Statistic", "p-value"]).set_index("Test").round(5)

def correlations(x, y, view_scatter = False):
    """
    Calculates and displays various correlation coefficients between two variables x and y.
    
    Parameters:
    - x (numpy.ndarray): The first variable for which correlation is to be calculated.
    - y (numpy.ndarray): The second variable for which correlation is to be calculated.
    - view_scatter (bool, optional): SFlag to display a scatter plot of the data points (default: False).
    
    Returns:
        pandas.DataFrame
    """


    methods = OrderedDict()
    methods["Pearson correlation coefficient"] = stats.pearsonr(x, y)
    methods["Spearman correlation coefficient"] = stats.spearmanr(x, y)
    methods["Kendall Tau"] = stats.kendalltau(x,y)
    methods["Weighted Kendall Tau"] = stats.weightedtau(x, y)

    res = []

    for i, (label, method) in enumerate(methods.items()):
        res.append([label, method[0], method[1]])

    if view_scatter ==True:
        sns.scatterplot(x, y)

    return pd.DataFrame(res, columns = ["Correlation", "Coefficient", "p-value"]).set_index("Correlation").round(5)








def contingency_table(col_1, col_2, alpha=0.05, print_expected_values=True):
    """
    Performs bivariate analysis on a set of ordinal data.
    
    Parameters
        col_1 (numpy.ndarray): Rows in the contingency table
        col_2 (numpy.ndarray): Columns in the contingency table
        alpha (float, optional): Value for test significance (default: 0.05)
        print_expected_values (bool, optional): Flag to returns expected values within the contingency table (default: True)
    
    Returns
        pandas.DataFrame: Contingency table

    """
    c_table = pd.crosstab(index=col_1, columns=col_2)
    chi2, p, dof, ex = chi2_contingency(c_table, correction=False)
    ex_int = ex.astype(int)
    
    n = np.sum(np.array(c_table))
    minDim = min(np.array(c_table).shape) - 1
    cramer_V = np.sqrt((chi2 / n) / minDim)
    
    if print_expected_values:
        for x, i in enumerate(c_table.columns):
            c_table[c_table.columns[x] + " expected"] = ex_int[:, x]
    
    c_table = c_table.reindex(sorted(c_table.columns), axis=1)
    
    if p < alpha:
        res = "The null hypothesis can be rejected"
    else:
        res = "The null hypothesis cannot be rejected"
    
    print(f"Chi squared: {chi2:.3f}")
    print(f"Degree of freedom: {dof}")
    print(f"p-value: {p:.6f} | {res}")
    print(f"Cramer's V: {cramer_V:.4f}")
    
    return c_table





def Complexity(x, info_selected, method = "procrustes", harmonic_start = 3, harmonic_stop = 5, 
    overall_score = True, dimensional_correction = True):
    """
    Calculates the complexity of given contour data using either the Procrustes distance or Hausdorff distance method.

    Args:
        x (aippa.utils.PotDataset or numpy.ndarray): Input data
        info_selected (pandas.DataFrame or pandas.Series): pandas.DataFrame or Series containing the index for the result DataFrame
        method (str, optional): Method to calculate complexity (default: “procrustes”). Possible values : “procrustes” or “hausdorff”
        harmonic_start (int, optional): Starting harmonic order (default: 3).
        harmonic_stop (int, optional): Ending harmonic order (default: 5)
        overall_score (bool, optional): Flag to calculate the mean complexity score (default: True)
        dimensional_correction (bool, optional): Flag to apply dimensional correction to the input data (default: True).

    Returns:
        pandas.DataFrame: DataFrame containing the complexity scores.

    Raises:
        ValueError: If an invalid method is selected.

    """

    complexity_list = []

    if dimensional_correction:
        trasformed_pots = []
        for pot in x[info_selected.index]:
            x = minimum_image(pot)
            new_img = transform.resize(x, (256,256))
            trasformed_pots.append(new_img)

        trasformed_array = np.array(trasformed_pots)
        cnt = ContourDescriptor(trasformed_array, verbose=False)
    else:
        cnt = ContourDescriptor(x[info_selected.index], verbose=False)

    for i, _ in enumerate(cnt):
        armonic_50 = pyefd.elliptic_fourier_descriptors(cnt[i], order=50, normalize=False)
        recostructed_50 = pyefd.reconstruct_contour(armonic_50, locus=(0, 0), num_points=300)

        complexity_n_list = []

        for n in range(harmonic_start, harmonic_stop+1):

            

            armonic_n = pyefd.elliptic_fourier_descriptors(cnt[i], order=n, normalize=False)
            recostructed_n = pyefd.reconstruct_contour(armonic_n, locus=(0, 0), num_points=300)
            
            if method == "procrustes":
                _, _ , proc_n= procrustes(recostructed_50, recostructed_n)
            
                proc_n = proc_n*1000

                complexity_n_list.append(proc_n)

            elif method == "hausdorff":
                
                h_d = directed_hausdorff(recostructed_50, recostructed_n)[0]

                complexity_n_list.append(h_d)

            else: 
                raise ValueError("Select 'procrustes' or 'hausdorff' as method")

        complexity_list.append(complexity_n_list)

    if method == "procrustes": 
        complexity_df = pd.DataFrame(complexity_list, columns=[f"procrustes_{r}" for r in range(harmonic_start, harmonic_stop+1)], index=info_selected.index)

    if method == "hausdorff":
        complexity_df = pd.DataFrame(complexity_list, columns=[f"hausdorff_{r}" for r in range(harmonic_start, harmonic_stop+1)], index=info_selected.index)
    
    if overall_score:
        complexity_df["score"] = complexity_df.mean(axis=1) 

    return complexity_df

def clustering_comparison(method, info, resampling_size = None, resampling_cyle = 25):
        """
        Perform a comparison analysis of clustering methods based on evaluation metrics.

        Parameters:
            method (object): The clustering method to evaluate. 
            info (pandas.DataFrame or pandas.Series): Ground truth information for the clustering. It should be a pandas DataFrame or Series containing the true labels for each data point.
            resampling_size (int, optional): The size of the resampled clusters. If not provided, the function evaluates the entire clusters. Default is None.
            resampling_cyle (int): The number of times resampling is performed. Default is 25.

        Returns:
            pandas.DataFrame: A DataFrame containing the evaluation scores for the clustering method.

        Raises:
            TypeError: If `resampling_size` is provided and not an integer.
            ValueError: If `resampling_size` is larger than the number of clusters.

        """
        if method.__class__.__name__ != "HierarchicalDBSCAN" and method.__class__.__name__ != "DensityBasedClustering":
            cluster_number = np.unique(method.cluster_labels).shape[0]
        else:
            no_noise_index = np.where(method.cluster_labels != -1)

            cluster_number = np.unique(method.cluster_labels[no_noise_index]).shape[0]
        
        if resampling_size:
            if type(resampling_size) != int:
                raise TypeError("resampling_size must be an integer")
            elif resampling_size > cluster_number:
                raise ValueError("resampling_size must be smaller than the number of clusters")
            else: 
           
                values = []
                
                for i in range(resampling_cyle):                
                    
                    selected_cluster = np.random.choice(cluster_number, size = resampling_size, replace = False)
                    selected_cluster_index = np.where(np.isin(method.cluster_labels, selected_cluster))
                    
                    adjusted_rand_score_value = adjusted_rand_score(info.values[selected_cluster_index], method.cluster_labels[selected_cluster_index])
                    adjusted_mutual_info_score_value = adjusted_mutual_info_score(info.values[selected_cluster_index], method.cluster_labels[selected_cluster_index])
                    homogeneity_score_value = homogeneity_score(info.values[selected_cluster_index], method.cluster_labels[selected_cluster_index])
                    completeness_score_value = completeness_score(info.values[selected_cluster_index], method.cluster_labels[selected_cluster_index])
                    v_measure_score_value = v_measure_score(info.values[selected_cluster_index], method.cluster_labels[selected_cluster_index])
                    fowlkes_mallows_score_value = fowlkes_mallows_score(info.values[selected_cluster_index], method.cluster_labels[selected_cluster_index])
                    
                    values.append([i, method.__class__.__name__, adjusted_rand_score_value, adjusted_mutual_info_score_value, homogeneity_score_value, completeness_score_value, v_measure_score_value, fowlkes_mallows_score_value, len(selected_cluster), resampling_cyle])    
                df = pd.DataFrame(values, columns=["Run", "Method", "Adjusted Rand score", "Adjusted Mutual Info score", "Homogeneity score", "Completeness score", "V-measure score", "Fowlkes Mallows score", "Number of clusters", "Resampling_cyle"]).set_index("Method")
                
                df = pd.DataFrame(df.mean()).T.drop(columns=["Run"])
                df.index = [method.__class__.__name__]
                return df
        else:
            selected_cluster = np.arange(cluster_number)
            selected_cluster_index = np.where(np.isin(method.cluster_labels, selected_cluster))
        
            values = []
            
            adjusted_rand_score_value = adjusted_rand_score(info.values[selected_cluster_index], method.cluster_labels[selected_cluster_index])
            adjusted_mutual_info_score_value = adjusted_mutual_info_score(info.values[selected_cluster_index], method.cluster_labels[selected_cluster_index])
            homogeneity_score_value = homogeneity_score(info.values[selected_cluster_index], method.cluster_labels[selected_cluster_index])
            completeness_score_value = completeness_score(info.values[selected_cluster_index], method.cluster_labels[selected_cluster_index])
            v_measure_score_value = v_measure_score(info.values[selected_cluster_index], method.cluster_labels[selected_cluster_index])
            fowlkes_mallows_score_value = fowlkes_mallows_score(info.values[selected_cluster_index], method.cluster_labels[selected_cluster_index])

            values.append([method.__class__.__name__, adjusted_rand_score_value, adjusted_mutual_info_score_value, homogeneity_score_value, completeness_score_value, v_measure_score_value, fowlkes_mallows_score_value, cluster_number])
            df = pd.DataFrame(values, columns=["Method", "Adjusted Rand score", "Adjusted Mutual Info score", "Homogeneity score", "Completeness score", "V-measure score", "Fowlkes Mallows score", "Number of clusters"]).set_index("Method")
            return df

