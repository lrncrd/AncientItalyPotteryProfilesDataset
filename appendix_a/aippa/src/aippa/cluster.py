import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from aippa.misc import SelBest, RemoveNoiseDBSCAN
import hdbscan
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
import plotly.figure_factory as ff

class Clustering_utils():
        
    def scatterplot(self, x=1, y = 2,  figsize = (8,8), savefig = False, legend_outside = False, plot_centroids = False, title = None, dpi = 300, **kwargs):
        """
        Create a scatter plot of the clustered data.

        This method generates a scatter plot of the clustered data based on the specified x and y features. The behavior of the method varies depending on the clustering algorithm used.

        Parameters:
            x (int, optional): 	Index of the x-axis component (default: 1).
            y (int, optional): Index of the y-axis component (default: 2)
            figsize (tuple, optional): Figure size (width, height) in inches (default: (10, 10)).
            savefig (bool, optional): Flag to save the plot as an image file (default: False).
            legend_outside (bool, optional): Flag to place the legend outside the plot (default: False).
            plot_centroids (bool, optional): Whether to plot centroids (available only for K_means and HierarchicalDBSCAN clustering) (default: False).
            title (str, optional):Title of the scatter plot (default: None).
            dpi (int, optional):DPI (dots per inch) for the saved image file (default: 300)
            **kwargs: Additional keyword arguments to be passed to the `sns.scatterplot` function (https://seaborn.pydata.org/generated/seaborn.scatterplot.html)..
        
        Returns:
            None

        Notes:
            Regarding the x and y parameters, each dimension is indexed starting from 1. For example, if I set x = 1 and y = 2, I'm currently use 0 and 1 indices in the reduced array.
        """
        if self.__class__.__name__ == "DensityBasedClustering" or self.__class__.__name__ == "HierarchicalDBSCAN":
            fig = plt.figure(figsize=figsize)
            noise_index = np.where(self.cluster_labels == -1)
            no_noise_index = np.where(self.cluster_labels != -1)
            
            if len(noise_index) > 0:
                ax = sns.scatterplot(x=self.data_for_plotting[:,(x-1)][noise_index], 
                y = self.data_for_plotting[:,(y-1)][noise_index], color = "grey", alpha = 0.8, marker="+")
            else:
                pass
            ax = sns.scatterplot(x=self.data_for_plotting[:,(x-1)][no_noise_index], 
            y = self.data_for_plotting[:,(y-1)][no_noise_index], hue = self.cluster_labels[no_noise_index],  **kwargs)
    
        else:
            fig = plt.figure(figsize=figsize)
            ax = sns.scatterplot(data=self.info_selected, x=self.data_for_plotting[:,(x-1)], 
            y = self.data_for_plotting[:,(y-1)], hue = self.cluster_labels, **kwargs)

        axis_name = "DIM"
        ax.set(xlabel=f'{axis_name} {x}', 
                ylabel=f'{axis_name} {y}')

        if (self.__class__.__name__ == "K_means" or self.__class__.__name__ == "HierarchicalDBSCAN") and plot_centroids:
            plt.scatter(x = self.cluster_centroid[:,0],  y = self.cluster_centroid[:,1], color = "red", edgecolor = "black", s = 50)   
        elif self.__class__.__name__ != "K_means" and plot_centroids:
            raise ValueError("Centroids are only available on 'K-means' and HDBSCAN clustering methods")


        if legend_outside:
            ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        if title:
            ax.set_title(title)
        if savefig:
            plt.savefig(f"{self.__class__.__name__}_Scatterplot.jpg", dpi = dpi, bbox_inches='tight')

        plt.show()
    
    def append_cluster_label(self):
        """
        Append cluster labels to the selected information DataFrame.

        Parameters:
            pandas.DataFrame: DataFrame with appended cluster labels column.
        
        Raises:
            Exception: If the info_selected attribute is not specified during the class initialization.
        """
        try:
            info_selected_appended = self.info_selected.copy() 
        except  AttributeError: 
            raise Exception("You must specify a DataFrame in which to save the cluster membership when creating the instance of the Class")
        info_selected_appended[f"{self.__class__.__name__}_cluster_labels"] = self.cluster_labels
        return info_selected_appended



class HierarchicalClustering():

    def __init__(self, x, info_selected = None, linkage = "ward", data_for_plotting= None, **kwargs):
        """
        HierarchicalClustering class performs hierarchical clustering on a given dataset.

        Parameters:
            x (numpy.ndarray): Input dataset for clustering.
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the data (default: None)
            linkage (str): Linkage criterion for hierarchical clustering (default: "ward").
                Available options: {"ward", "single", "complete", "average"}.
            data_for_plotting (numpy.ndarray): Dataset used for plotting purposes (default: None).
                If not provided, the input dataset 'x' will be used.
            **kwargs: Additional keyword arguments to be passed to the linkage function (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).

        Attributes:
            x (numpy.ndarray): Input dataset.
            data_for_plotting (numpy.ndarray): Dataset used for plotting purposes.
            clustering_method (str): Linkage criterion used for clustering.
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information related to the dataset.
            z (numpy.ndarray): The computed linkage matrix.
            cluster_labels (numpy.ndarray): Cluster labels assigned to each data point.

        Returns:
            HierarchicalClustering instance.
        """
        self.x = x
        if np.any(data_for_plotting):
            self.data_for_plotting = data_for_plotting
        else:
            self.data_for_plotting = x
            
        self.clustering_method = linkage
        self.info_selected = info_selected

        Z = hierarchy.linkage(self.x, linkage, **kwargs)

        self.z = Z
        self.cluster_labels = fcluster(self.z, 2, criterion='maxclust')

    def dendrogram(self, max_d = None, labels = None, figsize=(20, 10), savefig = False, dpi = 300, **kwargs):
        """
        Generate a dendrogram plot based on the computed hierarchical clustering.

        Parameters:
            max_d (float): The threshold value to color the dendrogram branches (default: None).
            labels (array_like): A list of labels corresponding to the data points (default: None)
            figsize (tuple): Figure size in inches (default: (20, 10)).
            savefig (bool): Whether to save the plot as an image (default: False).
            dpi (int): DPI (dots per inch) for the saved image file (default: 300).
            **kwargs: Additional keyword arguments to be passed to the dendrogram function (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html).

        Returns:
            None
        """
        
        fig = plt.figure(figsize=figsize)
        if max_d:
            if isinstance(labels, type(None)):
                dn = hierarchy.dendrogram(self.z, no_labels=True, color_threshold=max_d,**kwargs, )
                plt.axhline(y=max_d, c='k')
            else:
                dn = hierarchy.dendrogram(self.z, labels = labels, color_threshold=max_d,**kwargs, )
                plt.axhline(y=max_d, c='k')
        else:
            if isinstance(labels, type(None)):
                dn = hierarchy.dendrogram(self.z, no_labels=True, **kwargs)
            else:
                dn = hierarchy.dendrogram(self.z, labels = labels, **kwargs) 

        if savefig:
            
            plt.savefig(f"{self.__class__.__name__}_dendrogram.jpg", dpi = dpi, bbox_inches='tight')

        plt.show()
    
    def get_clusters(self, max_d = None):
        """
        Assign cluster labels to the data based on a given threshold or a specified number of clusters.

        Parameters:
            max_d (float): The threshold value to determine the clusters (default: None).

        Returns:
            numpy.ndarray: An array of cluster labels assigned to each data point.

        Note:
            - If 'max_d' is provided, clusters will be determined based on the distance criterion.
            - If 'max_d' is not provided, clusters will be determined based on the maximum number of clusters.
        """
        if max_d:
            self.cluster_labels = fcluster(self.z, max_d, criterion='distance')
            print("Cluster number set!")
        else:
            self.cluster_labels = fcluster(self.z, 2, criterion='maxclust')
        return self.cluster_labels

    def interactive_dendrogram(self, max_d=None, labels = None, **kwargs):
        """
        Generates an interactive dendrogram plot using Plotly.

        Parameters:
            max_d (float): The threshold value to determine the clusters (default: None).
            labels (str, list, or pandas.core.series.Series): Labels to display on the dendrogram (default: None).
            **kwargs: Additional keyword arguments to be passed to the Plotly dendrogram function (https://plotly.com/python/dendrogram/).

        Returns:
            None
        """
        if type(labels) == str:
            labels = self.info_selected[labels].tolist()
        elif type(labels) == list or type(labels) == pd.core.series.Series:
            labels = labels.tolist()
        elif type(labels) == np.ndarray:
            labels = labels.tolist()
        else:
            labels = None
        fig = ff.create_dendrogram(self.x, color_threshold = max_d, labels = labels, linkagefun=lambda x: hierarchy.linkage(x, self.clustering_method), **kwargs)
        fig.update_layout(width=1000, height=500)
        fig.show()
        

    def scatterplot(self, *args, **kwargs):
        Clustering_utils.scatterplot(self, *args, **kwargs)  

    def append_cluster_label(self, *args, **kwargs):
        return Clustering_utils.append_cluster_label(self, *args, **kwargs)       

        


class K_means(KMeans):

    def __init__(self, x, info_selected = None, n_clusters= 8, data_for_plotting = None, **kwargs):
        """
        A subclass of the scikit-learn KMeans class that provides additional functionalities.

        Parameters:
            x (numpy.ndarray): Input dataset for clustering.
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information associated with the data (default: None).
            n_clusters (int, optional): The number of clusters to create (default: 8).
            data_for_plotting (numpy.ndarray, optional): Dataset used for plotting purposes (default: None). If not provided, the input dataset ‘x’ will be used
            **kwargs: Additional keyword arguments to pass to the sklearn's KMeans class (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).

        Attributes:
            x (numpy.ndarray): Input dataset.
            data_for_plotting (numpy.ndarray): Data used for plotting purposes.
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information associated with the data.
            cluster_labels (numpy.ndarray): The cluster labels assigned to each data point.
            cluster (KMeans): The KMeans cluster object.
            cluster_centroid (numpy.ndarray): The coordinates of the cluster centroids.
        
        Returns:
            K_means instance.

        """

        self.x = x
        if np.any(data_for_plotting):
            self.data_for_plotting = data_for_plotting
        else:
            self.data_for_plotting = x
        cluster = KMeans(n_clusters = n_clusters, **kwargs).fit(self.x)

        if np.any(info_selected):
            self.info_selected = info_selected
        else:
            self.info_selected = None
        self.cluster_labels = cluster.labels_
        self.cluster = cluster
        self.cluster_centroid = cluster.cluster_centers_


            

    def elbow_plot(self, figsize = (10,5), k_range = (1,15), savefig = False, dpi = 300):
        """
        Generates an elbow plot to determine the optimal number of clusters.

        Parameters:
            figsize (tuple, optional): Figure size (width, height) in inches (default: (10, 5)).
            k_range (tuple, optional): Range of k values to test (default: (1, 15)).
            savefig (bool, optional): Flag to save the plot as an image file (default: False).
            dpi (int, optional): DPI (dots per inch) for the saved image file (default: 300).

        Returns:
            None

        """

        plt.figure(figsize=figsize)
        Sum_of_squared_distances = []
        K = range(k_range[0],k_range[1])
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(self.x)
            Sum_of_squared_distances.append(km.inertia_)
        plt.plot(K, Sum_of_squared_distances, 'o--')
        plt.xlabel('k')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow Method For Optimal k')
        plt.xticks(K)
        if savefig:
            plt.savefig(f"{self.__class__.__name__}_Elbow_plot.jpg", dpi = dpi)


    def scatterplot(self, *args, **kwargs):

        Clustering_utils.scatterplot(self, *args, **kwargs)
    
    def append_cluster_label(self, *args, **kwargs):
        return Clustering_utils.append_cluster_label(self, *args, **kwargs)  


class DensityBasedClustering(DBSCAN):
    
    def __init__(self, x, info_selected = None, eps=0.5, min_samples = 5, data_for_plotting = None, verbose = True, **kwargs):
        """
        Density-based clustering using the DBSCAN algorithm.

        Parameters:
            x (numpy.ndarray): Input dataset for clustering.
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information associated with the data (default: None).
            eps (float, optional): The maximum distance between two samples for them to be considered as in the same neighborhood (default: 0.5).
            min_samples (int, optional): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point (default: 5).
            data_for_plotting (numpy.ndarray, optional): Data used for plotting (default: None).
            verbose (bool, optional): Flag to enable/disable verbose. A range of information are printed (default: True).
            **kwargs: Additional keyword arguments to be passed to the sklearn's DBSCAN algorithm (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

        Attributes:
            x (numpy.ndarray): Input dataset
            data_for_plotting (numpy.ndarray): 	Dataset used for plotting purposes.
            info_selected (pandas.Dataframe or pandas.Series): Additional information associated with the data.
            cluster_labels (numpy.ndarray): Cluster labels assigned to each data point.
            cluster (DBSCAN): DBSCAN clustering object.
            cluster_number (int): Number of clusters found.
            noise_number (int): Number of noise points.
        
        Returns:
            DensityBasedClustering instance.

        """
        self.x = x
        if np.any(data_for_plotting):
            self.data_for_plotting = data_for_plotting
        else:
            self.data_for_plotting = x
        cluster = DBSCAN(eps=eps, min_samples=min_samples, **kwargs).fit(self.x)
        if np.any(info_selected):
            self.info_selected = info_selected
        else:
            self.info_selected = None
        self.cluster_labels = cluster.labels_
        self.cluster = cluster
        self.info_selected = info_selected
        cluster_number = (max(self.cluster_labels) + 1)

        noise_number = len(self.cluster_labels[self.cluster_labels == -1])
        

        if verbose:
            print(f"Using these parameters DBSCAN (eps:{eps}; min_sample:{min_samples}) finds {cluster_number} clusters. {noise_number} noise-points")


    def k_plot(self, figsize = (10,5), savefig = False, dpi = 300, **kwargs):
        """
        Plot the k-distance graph.

        Parameters:
            figsize (tuple, optional): Figure size (width, height) in inches (default: (10, 5)).
            savefig (bool, optional): Flag to save the plot as an image file (default: False).
            dpi (int, optional): DPI (dots per inch) for the saved image file (default: 300).
            **kwargs: Additional keyword arguments to be passed to the matplotlib's plot function.

        Returns:
            None

        """
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(self.x)
        distances, _ = nbrs.kneighbors(self.x)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        plt.figure(figsize=figsize)
        plt.plot(distances, **kwargs)
        plt.title('K-distance Graph')
        plt.xlabel('Data Points sorted by distance')
        plt.ylabel('Epsilon')
        plt.show()
        if savefig:
            plt.savefig(f"{self.__class__.__name__}_K_distance_plot.jpg", dpi = dpi, bbox_inches='tight')

    def scatterplot(self, *args, **kwargs):
        Clustering_utils.scatterplot(self, *args, **kwargs)

    def append_cluster_label(self, *args, **kwargs):
        return Clustering_utils.append_cluster_label(self, *args, **kwargs)  

class GaussianMixtureClustering(GaussianMixture):
    
    def __init__(self, x, info_selected = None, n_components = 3, data_for_plotting = None, **kwargs):
        """
        Gaussian Mixture Model-based clustering.

        Args:
            x (numpy.ndarray): Input dataset for clustering.
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the data (default: None).
            n_components (int, optional): Number of mixture components (clusters) to fit (default: 3).
            data_for_plotting (numpy.ndarray, optional): Data used for plotting or visualization (default: None).
            **kwargs: Additional keyword arguments to be passed to the sklearn's GaussianMixture estimator (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture).

        Attributes:
            x (numpy.ndarray): Input data.
            data_for_plotting (numpy.ndarray): Data used for plotting or visualization.
            info_selected (pandas.Dataframe or pandas.Series): Additional information or labels associated with the data.
            cluster_labels (numpy.ndarray): Cluster labels assigned to each data point.
            cluster (GaussianMixture): GaussianMixture model fitted to the data.

            Returns:
                GaussianMixtureClustering instance.
        """
        self.x = x
        if np.any(data_for_plotting):
            self.data_for_plotting = data_for_plotting
        else:
            self.data_for_plotting = x
        cluster = GaussianMixture(n_components=n_components, **kwargs).fit(self.x)
        if np.any(info_selected):
            self.info_selected = info_selected
        else:
            self.info_selected = None
        self.cluster_labels = cluster.predict(x)
        self.cluster = cluster




    def BIC_AIC_scores_plot(self, n_iterations = 10, max_cluster_numbers = 20, dpi = 300, figsize= (10,5), savefig = False):
        """
        Plot the BIC and AIC scores for different numbers of clusters.

        Parameters:
            n_iterations (int, optional): Number of iterations for fitting the Gaussian Mixture Model (default: 10).
            max_cluster_numbers (int, optional): Maximum number of clusters to consider (default: 20).
            figsize (tuple, optional): Figure size (width, height) in inches (default: (10, 5)).
            savefig (bool, optional): Flag to save the plot as an image file (default: False).
            dpi (int, optional): DPI (dots per inch) for the saved image file (default: 300).
        Returns:
            None

        """
        n_clusters=np.arange(2, max_cluster_numbers+1)
        bics=[]
        aics = []
        bics_err=[]
        aics_err = []
        iterations=n_iterations
        for n in n_clusters:
            tmp_bic=[]
            tmp_aic = []
            for _ in range(iterations):
                gmm=GaussianMixture(n, n_init=2).fit(self.x) 
                
                tmp_bic.append(gmm.bic(self.x))
                tmp_aic.append(gmm.aic(self.x))
            val_bic=np.mean(SelBest(np.array(tmp_bic), int(iterations/5)))
            err_bic=np.std(tmp_bic)
            val_aic=np.mean(SelBest(np.array(tmp_aic), int(iterations/5)))
            err_aic=np.std(tmp_aic)
            bics.append(val_bic)
            bics_err.append(err_bic)
            aics.append(val_aic)
            aics_err.append(err_aic)

        
        fig, ax = plt.subplots(1,1, figsize=figsize)
        
        ax.set_title("BIC and AIC Scores ", fontsize=15)
        ax.errorbar(n_clusters,bics, yerr=bics_err, label='BIC')
        ax.errorbar(n_clusters,aics, yerr=aics_err, label='AIC')
        ax.set_xlim(1, max(n_clusters)+1)
        ax.set_xticks(np.arange(min(n_clusters), max(n_clusters)+1, 1.0))
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Score")
        ax.legend()


        if savefig:
            plt.savefig(f"{self.__class__.__name__}_BIC_AIC_scoreplot.jpg", dpi = dpi, bbox_inches='tight')

    plt.show()

    def get_new_samples(self, number = 1000):
        """
        Generate new samples from the fitted GaussianMixture Model.

        Parameters:
            number (int, optional): Number of new samples to generate (default: 1000).

        Returns:
            new_values (numpy.ndarray): Array of shape (number, n_features) containing the generated samples.
            new_labels (numpy.ndarray): Array of shape (number,) containing the corresponding cluster labels for the generated samples.

        """

        new_data = self.cluster.sample(number)
        new_values = new_data[0]
        new_labels = new_data[1]
        return new_values, new_labels

    def scatterplot(self, *args, **kwargs):
        Clustering_utils.scatterplot(self, *args, **kwargs)
    
    def append_cluster_label(self, *args, **kwargs):
        return Clustering_utils.append_cluster_label(self, *args, **kwargs)  

class HierarchicalDBSCAN(hdbscan.HDBSCAN):

    def __init__(self, x, info_selected = None, min_cluster_size =5, min_samples = None, data_for_plotting = None, verbose = True, **kwargs):
        """
        Hierarchical DBSCAN clustering algorithm.

        Parameters:
            x (numpy.ndarray): Input dataset for clustering.
            info_selected (pandas.Dataframe or pandas.Series): Additional information or labels associated with the data (default: None)
            min_cluster_size (int, optional): The minimum size of clusters (default: 5).
            min_samples (int, optional): The number of samples in a neighbourhood for a point to be considered a core point (default: None).
            data_for_plotting (numpy.ndarray, optional): Dataset used for plotting purposes (default: None). If not provided, the input dataset ‘x’ will be used
            verbose (bool, optional):Flag to enable/disable verbose. A range of information are printed (default: True).
            **kwargs: Additional keyword arguments to be passed to hdbscan.HDBSCAN (https://hdbscan.readthedocs.io/en/latest/parameter_selection.html).

        Attributes:
            x (numpy.ndarray): Input dataset
            data_for_plotting (numpy.ndarray): Dataset used for plotting purposes.
            info_selected (pandas.Dataframe or pandas.Series): Additional information or labels associated with the data
            cluster_labels (numpy.ndarray): Cluster labels assigned to each data point.
            cluster (hdbscan.HDBSCAN): HDBSCAN clustering object.
            cluster_number (int): Number of clusters found.
            cluster_centroid (numpy.ndarray): Array of shape (n_clusters, n_features) containing the cluster centroids.

        Returns:
            HierarchicalDBSCAN instance.
            
        """
        self.x = x
        if np.any(data_for_plotting):
            self.data_for_plotting = data_for_plotting
        else:
            self.data_for_plotting = x
        cluster = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, min_samples = min_samples, **kwargs).fit(self.x)
        self.x = x
        self.cluster_labels = cluster.labels_
        self.cluster = cluster
        self.info_selected = info_selected
        cluster_number = (max(self.cluster_labels) + 1)
        if cluster_number >= 3:
            cluster_centroid = []
            pipeline_clusters = self.cluster_labels[self.cluster_labels != -1]
            for i in range(max(pipeline_clusters)+1):
                cluster_centroid.append(self.cluster.weighted_cluster_centroid(i))
            self.cluster_centroid = np.array(cluster_centroid)
        else:
            self.cluster_centroid = None
            

        noise_number = len(self.cluster_labels[self.cluster_labels == -1])
        if verbose:
            print(f"Using these parameters HDBSCAN finds {cluster_number} clusters. {noise_number} noise-points")



        
    def condensed_tree_plot(self, select_clusters=False, figsize = (15,8), savefig = False, dpi = 300):
        """
        Plot the condensed tree of the hierarchical DBSCAN clustering.

        Parameters:
            select_clusters (bool, optional): Flag to highlight selected clusters (default: False).
            figsize (tuple, optional): Figure size (width, height) in inches (default: (15, 8)).
            savefig (bool, optional): Flag to save the plot as an image file (default: False).
            dpi (int, optional): DPI (dots per inch) for the saved image file (default: 300).
        """
        plt.figure(figsize=figsize)
        self.cluster.condensed_tree_.plot(select_clusters = select_clusters)
        if savefig:
            plt.savefig("condensed_tree.jpg", dpi = dpi, bbox_inches='tight')  
    
    def scatterplot(self, *args, **kwargs):
        Clustering_utils.scatterplot(self, *args, **kwargs)
    
    def append_cluster_label(self, *args, **kwargs):
        return Clustering_utils.append_cluster_label(self, *args, **kwargs) 