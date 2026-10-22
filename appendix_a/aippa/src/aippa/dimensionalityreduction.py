import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn import manifold
import umap
import umap.plot
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances



class Dimension_reduction_utils():

    def scatterplot(self, x=1, y = 2,  figsize = (8,8), savefig = False, legend_outside = False, title = None, dpi = 300, **kwargs):
        """
        Generates a scatter plot based on the reduced data.

        Args:
            x (int, optional): Index of the x-axis component. Defaults to 1.
            y (int, optional): Index of the y-axis component. Defaults to 2.
            figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (10, 10).
            savefig (bool, optional): Flag to save the plot as an image file. Defaults to False.
            legend_outside (bool, optional): Flag to place the legend outside the plot. Defaults to False.
            title (str, optional): Title of the scatter plot. Defaults to None.
            dpi (int, optional): DPI (dots per inch) for the saved image file. Defaults to 300.
            **kwargs: Additional keyword arguments to be passed to the `sns.scatterplot` function (https://seaborn.pydata.org/generated/seaborn.scatterplot.html).
        
        Returns:
            None

        Description:
            This method generates a scatter plot of the selected data, where each point represents an instance in the data.
            The x-axis and y-axis values are determined by the indices provided in the 'x' and 'y' parameters.

            Depending on the class type of the instance, the method customizes the labels for the axes. If the instance belongs
            to the PrincipalComponentAnalysis (PCA) or LinearDiscriminantAnalysis (LDA) classes, the axes are labeled as PC (Principal
            Component) with their corresponding explained variance ratios. If the instance belongs to the T_SNE or Umap classes, the
            axes are labeled as EMB (Embedding). If the instance belongs to the MultidimensionalScaling (MDS) class, the axes are
            labeled as MDS.

            If the 'legend_outside' parameter is set to True, the legend is placed outside the plot. If the 'title' parameter is provided,
            it sets the title of the scatter plot.

            If the 'savefig' parameter is set to True, the generated plot is saved as an image file named "{class_name}_Scatterplot.jpg"
            in the current directory.

        Notes:
            Regarding the x and y parameters, each dimension is indexed starting from 1. For example, if I set x = 1 and y = 2, I'm currently use 0 and 1 indices in the reduced array.

        """
        fig = plt.figure(figsize=figsize)
        ax = sns.scatterplot(data=self.info_selected, x=self.reduction[:,(x-1)], y = self.reduction[:,(y-1)], **kwargs)

        if self.__class__.__name__ == "PrincipalComponentAnalysis" or self.__class__.__name__ == "LDA":
            axis_name = "PC"
            ax.set(xlabel=f'{axis_name} {x} \n {self.method.explained_variance_ratio_[x-1]:.3f} %', 
                ylabel=f'{axis_name} {y} \n {self.method.explained_variance_ratio_[y-1]:.3f} %') 
        elif self.__class__.__name__ == "T_SNE" or self.__class__.__name__ == "Umap":
            axis_name = "EMB"
            ax.set(xlabel=f'{axis_name} {x}', 
                ylabel=f'{axis_name} {y}')
        elif  self.__class__.__name__ == "MultidimensionalScaling":
            axis_name = "MDS"
            ax.set(xlabel=f'{axis_name} {x}', 
                ylabel=f'{axis_name} {y}')
        else:
            pass

        if legend_outside:
            ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
        if title:
            ax.set_title(title)
        if savefig:
            plt.savefig(f"{self.__class__.__name__}_Scatterplot.jpg", dpi = dpi, bbox_inches='tight')

        plt.show()


class PrincipalComponentAnalysis(decomposition.PCA):

    def __init__(self, x, n_components=2, info_selected=None, verbose=True, **kwargs):
        """
        Principal Component Analysis (PCA) class for dimensionality reduction.

        Parameters:
            x (numpy.ndarray): Input data matrix.
            n_components (int, optional): Number of principal components (default: 2).
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the data (default: None).
            verbose (bool, optional): Flag to enable/disable verbose (default: True).
            **kwargs: Additional keyword arguments to be passed to sklearn's PCA algorithm (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

        Attributes:
            x (numpy.ndarray): Input data matrix.
            info_selected (pandas.Dataframe or pandas.Series): Additional information or labels associated with the data.
            reduction (numpy.ndarray): Reduced data matrix after applying PCA.
            method (PCA): sklearn's PCA instance.

        Returns:
            PrincipalComponentAnalysis instance.
        """
        self.x = x
        self.info_selected = info_selected
        pca = decomposition.PCA(n_components=n_components, **kwargs).fit(x)
        self.reduction = pca.transform(x)
        self.method = pca
        
        if verbose:
            print(f"Original dimension: {self.x.shape}")
            print(f"Output dimension: {self.reduction.shape}")
            print(f"First 5 components variance: {np.round(self.method.explained_variance_ratio_[:5], 2)}")

    def scatterplot(self, *args, **kwargs):

        Dimension_reduction_utils.scatterplot(self, *args, **kwargs)
        
    def explained_variance_cumulative_plot(self, figsize = (10,5), savefig = False):
        """
        Generates a cumulative explained variance plot based on the PCA results.

        Parameters:
            figsize (tuple, optional): Figure size (width, height) in inches (default: (10,5)).
            savefig (bool, optional): Flag to save the plot as an image as “PCA_Explained_variance_cumulative_plot.jpg” (default: False).

        """
        fig = plt.figure(figsize=figsize)
        sns.lineplot(y = np.cumsum(self.method.explained_variance_ratio_), x = range(1, len(self.method.explained_variance_ratio_)+1), marker = "o")
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.title('Cumulative explained variance ratio as a function of the number of components')
        plt.xticks(np.arange(1, len(self.method.explained_variance_ratio_)+1, 1.0))
        plt.tight_layout()
       
        
        if savefig:
            plt.savefig("Explained_variance_cumulative_plot.jpg", dpi = 300)

        plt.show()

    def explained_variance_plot(self, figsize = (10,5), savefig = False):
        """
        Generates a bar plot to visualize the explained variance ratio of the principal components.

        Args:
            figsize (tuple, optional): Figure size (width, height) in inches (default: (10,5)).
            savefig (bool, optional): Flag to save the plot as an image as “Explained_variance_plot.jpg” (default: False).
        """
        fig = plt.figure(figsize=figsize)
        sns.barplot(x = np.arange(len(self.method.explained_variance_ratio_)), y = self.method.explained_variance_ratio_)
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.title('Explained variance ratio of the principal components')
        plt.tight_layout()
        
        if savefig:
            plt.savefig("Explained_variance_plot.jpg", dpi = 300)
        
        plt.show()

    def plot_eigen_pots(self, savefig=False):
        """
        Plots the eigen-pots of the principal components.

        Args:
            savefig (bool, optional): Flag to save the plot as an image file as “Eigen_pots_plot.jpg” (default: False).

        Raises:
            ValueError: If the features are not of type `numpy.ndarray`.
            ValueError: If the features matrix is not a squared matrix.
        """
        if not isinstance(self.x, np.ndarray):
            raise ValueError("PlotEigenPots needs numpy.ndarray as features")

        if self.x[0].shape != np.sqrt(self.x[0].shape)**2:
            raise ValueError("PlotEigenPots needs squared matrix (use Vectorization as features extraction method)")

        shape = int(np.sqrt(self.x[0].shape))
        num_components = len(self.method.explained_variance_ratio_)
        num_cols = int(np.sqrt(num_components))
        num_rows = int(np.ceil(num_components / num_cols))

        fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6, 3*num_rows))

        for i, n in enumerate(self.method.components_):
            if i >= num_components:
                break

            ax.flatten()[i].imshow(n.reshape(shape, shape), cmap='viridis')
            ax.flatten()[i].set_title(f"Component {i+1}")

        for j in range(num_components, num_rows*num_cols):
            ax.flatten()[j].axis('off')

            plt.tight_layout()

        if savefig:
            plt.savefig("Eigen_pots_plot.jpg", dpi=300)

        plt.show()





class TruncSVD(decomposition.TruncatedSVD):
    """
    Dimensionality reduction using truncated singular value decomposition (SVD).

    Parameters:
        x (numpy.ndarray): Input data matrix.
        n_components (int, optional): Number of principal components (default: 2).
        info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the data (default: None).
        verbosity (bool, optional): Flag to enable/disable verbose. A range of information are printed (default: True).
        **kwargs: Additional keyword arguments to be passed to sklearn's TruncatedSVD algorithm (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).
    
    Attributes:
        x (numpy.ndarray): Input data matrix.
        info_selected (pandas.Dataframe or pandas.Series): Additional information or labels associated with the data.
        reduction (ndarray): Reduced data matrix after applying PCA.
        method (TruncatedSVD): sklearn's TruncatedSVD instance.

    Returns:
        TruncSVD instance.
    """
    def __init__(self, to_reduce,n_components = 2, info_selected = None, verbose= False, **kwargs):
        self.to_reduce = to_reduce
        self.info_selected = info_selected
        svd = decomposition.TruncatedSVD(n_components= n_components, **kwargs)
        X = svd.fit_transform(to_reduce)
        self.reduction = X
        self.method = svd


        if verbose:
            print(f"Original dimension: {self.to_reduce.shape}")
            print(f"Output dimension: {self.reduction.shape}")
            print(f"First 5 components variance: {np.round(self.method.explained_variance_ratio_[:5], 2)}")

    def scatterplot(self, *args, **kwargs):

        Dimension_reduction_utils.scatterplot(self, *args, **kwargs)
    
    def explained_variance_cumulative_plot(self):
        """
        Generates a cumulative explained variance plot based on the PCA results.

        Parameters:
            figsize (tuple, optional): Figure size (width, height) in inches (default: (10,5)).
            savefig (bool, optional): Flag to save the plot as an image as “PCA_Explained_variance_cumulative_plot.jpg” (default: False).

        """
        PrincipalComponentAnalysis.explained_variance_cumulative_plot(self)
    
    def explained_variance_plot(self):
        """
        Generates a bar plot to visualize the explained variance ratio of the principal components.

        Args:
            figsize (tuple, optional): Figure size (width, height) in inches (default: (10,5)).
            savefig (bool, optional): Flag to save the plot as an image as “Explained_variance_plot.jpg” (default: False).
        """
        PrincipalComponentAnalysis.explained_variance_plot(self)

####### Manifold ##########

class MultidimensionalScaling(manifold.MDS):

    
    def __init__(self, x, n_components = 2, info_selected = None, **kwargs):
        """
        Dimensionality reduction using Multidimensional Scaling (MDS).

        Parameters:
            x (numpy.ndarray): The input data matrix.
            n_components (int, optional): The number of dimensions in which to project the data (default: 2).
            info_selected (pandas.Dataframe or pandas.Series, optional):Additional information or labels associated with the data (defaults: None).
            **kwargs: Additional keyword arguments to be passed to the `manifold.MDS` class (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html).

        Attributes:
            x (numpy.ndarray): Input data.
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the data.
            reduction (numpy.ndarray): The reduced representation of the input data.
            method (manifold.MDS): sklearn's manifold.MDS instance.
        
        Returns:
            MultidimensionalScaling instance.

        """
        self.x = x
        self.info_selected = info_selected
        mds = manifold.MDS(n_components=n_components, **kwargs)
        X = mds.fit_transform(x)
        self.reduction = X
        self.method = mds
        
    def scatterplot(self, *args, **kwargs):

        Dimension_reduction_utils.scatterplot(self, *args, **kwargs)

class t_SNE(manifold.TSNE):
    def __init__(self, x, n_components = 2, info_selected = None, **kwargs):
        """
        Dimensionality reduction using T-distributed Stochastic Neighbor Embedding (TSNE).

        Parameters:
            x (numpy.ndarray): The input data matrix.
            n_components (int, optional): The number of dimensions in which to project the data (default: 2).
            info_selected (pandas.Dataframe or pandas.Series, optional):Additional information or labels associated with the data (defaults: None).
            **kwargs: Additional keyword arguments to be passed to the `manifold.TSNE` class (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE).

        Attributes:
            x (numpy.ndarray): The input data array or matrix.
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the data.
            reduction (numpy.ndarray): The reduced representation of the input data.
            method (manifold.TSNE): sklearn's manifold.TSNE instance.
        
        Returns:
            t_SNE instance.

        """
        
        self.x = x
        self.info_selected = info_selected
        tsne = manifold.TSNE(n_components=n_components, **kwargs)
        X = tsne.fit_transform(x)
        self.reduction = X
        self.method = tsne

    def scatterplot(self, *args, **kwargs):

        Dimension_reduction_utils.scatterplot(self, *args, **kwargs)

class AippaUmap(umap.UMAP):

    def __init__(self, x, n_components = 2, info_selected = None, **kwargs):
        """
        Dimensionality reduction using Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP). 

        Parameters:
            x (numpy.ndarray): The input data matrix.
            n_components (int, optional): The number of dimensions in which to project the data (default: 2).
            info_selected (pandas.Dataframe or pandas.Series, optional):Additional information or labels associated with the data (defaults: None).
            **kwargs: Additional keyword arguments to be passed to the `umap.UMAP` class (https://umap-learn.readthedocs.io/en/latest/).

        Attributes:
            x (numpy.ndarray): The input data array or matrix.
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the data.
            reduction (numpy.ndarray): The reduced representation of the input data.
            method (umap.UMAP): umap.UMAP instance.

        Returns:
            AippaUmap instance.

        Notes:
            The class has this name to distinguish it from the original implementation (UMAP or umap).
        """
        self.x = x
        self.info_selected = info_selected
        mapper = umap.UMAP(n_components=n_components, **kwargs)
        X = mapper.fit_transform(x)
        self.reduction = X
        self.method = mapper

    def scatterplot(self, *args, **kwargs):

        Dimension_reduction_utils.scatterplot(self, *args, **kwargs)

    def show_connectivity(self, show_points = True, edge_bundling=None):
        """
        Displays the connectivity graph of the UMAP embedding, if the data is 2-dimensional.

        Parameters:
            show_points (bool, optional): Flag to show the points in the plot (default: True).
            edge_bundling (str, optional): Type of edge bundling to apply. Defaults to None. Possible values: None, "hammer".
        """
        if self.reduction.shape[1] == 2:
            umap.plot.connectivity(self.method, show_points=show_points, edge_bundling = edge_bundling)
            plt.show()
        else:
            print("This method is only available for 2D data: Reinstanciate the class with n_components = 2")
        
############  Supervised Dimension Reduction ##############


class LDA(LinearDiscriminantAnalysis):
    

    def __init__(self, x, y, n_components = 2, info_selected = None, verbose = True, **kwargs):
        """
        Supervised dimensionality reduction technique using Linear Discriminant Analysis (LDA).

        Parameters:
            x (numpy.ndarray): Input data matrix.
            y (numpy.ndarray): The target variable for supervised dimensionality reduction.
            n_components (int, optional): The number of dimensions in which to project the data (default: 2).
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the data (default: None).
            verbose (bool, optional): Flag to enable/disable verbose. A range of information are printed (default: True).
            **kwargs: Additional keyword arguments to be passed to the `LinearDiscriminantAnalysis` class (https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

        Attributes:
            x (numpy.ndarray): The input data array or matrix.
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the reduced data.
            reduction (numpy.ndarray): The reduced data.
            y (numpy.ndarray): The target variable for supervised dimensionality reduction.
            method (LinearDiscriminantAnalysis): The sklearn's LinearDiscriminantAnalysis instance.

        Returns:
            LinearDiscriminantAnalysis instance.

        Notes:
            y argument will be encoded using sklearn's LabelEncoder (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).

        """
        self.x = x
        self.info_selected = info_selected
        lda = LinearDiscriminantAnalysis(n_components= n_components, **kwargs)
        X = lda.fit_transform(x, LabelEncoder().fit_transform(y))
        self.reduction = X
        self.y = y
        self.method = lda

        if verbose:
            print(f"Original dimension: {self.x.shape}")
            print((f"Output dimension: {self.reduction.shape}"))
            print((f"First 5 components variance: {self.method.explained_variance_ratio_[:5]}"))

    def scatterplot(self, *args, **kwargs):
        Dimension_reduction_utils.scatterplot(self, *args, **kwargs)
    
    def explained_variance_cumulative_plot(self, *args, **kwargs):
        PrincipalComponentAnalysis.explained_variance_cumulative_plot(self, *args, **kwargs)
    
    def explained_variance_plot(self, *args, **kwargs):
        PrincipalComponentAnalysis.explained_variance_plot(self, *args, **kwargs)




class SupervisedUmap(AippaUmap):
    def __init__(self, x, y, n_components = 2, info_selected = None, **kwargs):
        """
        Supervised UMAP (Uniform Manifold Approximation and Projection) dimensionality reduction technique.

        Parameters:
            x (numpy.ndarray): Input data matrix.
            y (numpy.ndarray): The target variable for supervised dimensionality reduction.
            n_components (int, optional): The number of dimensions in which to project the data (default: 2).
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the data (default: None).
            **kwargs: Additional keyword arguments to be passed to the `umap.UMAP` class (https://umap-learn.readthedocs.io/en/latest/).

        Attributes:
            x (numpy.ndarray): The input data array or matrix.
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the data.
            reduction (numpy.ndarray): The reduced representation of the input data.
            method (umap.UMAP): umap.UMAP instance.

        Returns:
            SupervisedUmap instance.

        Notes:
            y argument will be encoded using sklearn's LabelEncoder (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).
        """

        self.x = x
        le = LabelEncoder() 
        self.info_selected = info_selected
        X = umap.UMAP(n_components=n_components,  **kwargs).fit_transform(x, le.fit_transform(y))
        mapper = umap.UMAP(**kwargs).fit(x, le.fit_transform(y))
        self.reduction = X
        self.method = mapper


class NCA(NeighborhoodComponentsAnalysis):

    def __init__(self, x, y, n_components=None, info_selected = None, init="auto", **kwargs):
        """
        Neighborhood Components Analysis (NCA)supervised dimensionality reduction technique.

        Parameters:
            xx (numpy.ndarray): Input data matrix.
            y (numpy.ndarray): The target variable for supervised dimensionality reduction.
            n_components (int, optional): The number of dimensions in which to project the data (default: 2).
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the data (default: None).
            init (str, optional): Initialization of the linear transformation. Possible options are 'auto', 'pca', 'identity', and 'random' (default: 'auto').
            **kwargs: Additional keyword arguments to be passed to the `NeighborhoodComponentsAnalysis` class (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html).

        Attributes:
            x (numpy.ndarray): The input data array or matrix.
            info_selected (pandas.Dataframe or pandas.Series, optional): Additional information or labels associated with the data.
            reduction (numpy.ndarray): The reduced representation of the input data.
            method (NeighborhoodComponentsAnalysis): NeighborhoodComponentsAnalysis instance.

        Returns:
            NCA instance.

        Notes:
            y argument will be encoded using sklearn's LabelEncoder (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).
        """
        nca = NeighborhoodComponentsAnalysis(n_components, init=init, **kwargs).fit(x, y)
        self.info_selected = info_selected
        self.method =  nca
        self.reduction = nca.fit_transform(x, y)

    def scatterplot(self, *args, **kwargs):
        Dimension_reduction_utils.scatterplot(self, *args, **kwargs)
