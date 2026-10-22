from skimage.feature import hog
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import measure, feature
import pyefd
from aippa.misc import interpcurve


class HOGDescriptor:

    def __init__(self, x, sigma=5, orientations=8, pixels_per_cell=(8, 8), cell_per_block=(1, 1), verbose=True):
        """
        HOGDescriptor class computes Histogram of Oriented Gradients (HOG) features for a set of images.

        Parameters:
            x (aippa.utils.PotDataset or numpy.ndarray): IInput data to extract vector representations from. If x is of type aippa.utils.PotDataset, 
                it will be copied to x. If x is of type numpy.ndarray and has 2 dimensions, it will be assigned to x directly. Otherwise, x will be assigned as a NumPy array of x
            sigma (float): Standard deviation for the Canny edge detector (default: 5).
            orientations (int): Number of orientation bins for the histograms (default: 8).
            pixels_per_cell (tuple): Number of pixels in each cell (default: (8, 8)).
            cell_per_block (tuple): Number of cells in each block (default: (1, 1)).
            verbose (bool): Flag to display progress bar (default: True).

        Attributes:
            x (numpy.ndarray): Input images to compute HOG features.
            images (list): List of HOG images computed for each input image.
            hogs (list): List of HOG feature vectors computed for each input image.
            values (numpy.ndarray): HOG feature vectors as a numpy array.
        
        Returns:
            HOGDescriptor instance.

        Note:
            The HOG features are computed using the skimage.feature.hog function, for more information see: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html.
            For more information about the Canny edge detector, see: https://scikit-image.org/docs/dev/auto_examples/edges/plot_canny.html.
            
        """
        if type(x) == "aippa.utils.PotDataset":
            self.x = x[:]
        elif type(x) == "numpy.ndarray" and len(x.shape) == 2:
            self.x = x
        else:
            self.x = np.array(x)

        self.images = []
        self.hogs = []
        if verbose:
            for elem in tqdm(x, "Computing HOGs"):
                x = feature.canny(elem, sigma=sigma)
                hog_v, hog_image = hog(
                    x, orientations=orientations,
                    pixels_per_cell=pixels_per_cell, cells_per_block=cell_per_block,
                    visualize=True
                )
                self.images.append(hog_image)
                self.hogs.append(hog_v)
        else:
            for elem in x:
                x = feature.canny(elem, sigma=sigma)
                hog_v, hog_image = hog(
                    x, orientations=orientations,
                    pixels_per_cell=pixels_per_cell, cells_per_block=cell_per_block,
                    visualize=True
                )
                self.images.append(hog_image)
                self.hogs.append(hog_v)
        self.values = np.array(self.hogs)

    def show_example(self, seed=0, n_examples=4, savefig=False):
        """
        Displays examples of the original images and their corresponding HOG feature visualizations.

        Parameters:
            seed (int): Starting index of the examples to display (default: 0).
            n_examples (int): Number of examples to display (default: 4).
            savefig (bool): Flag to save the plot as an image file as "hogs_example.jpg" (default: False).
        """
        assert seed < self.__len__(), "Seed is too large, use a smaller seed."
        assert seed >= 0, "Cannot use a seed less than 0."
        if seed < self.__len__() < (seed + n_examples):
            n_examples = self.__len__() - seed
        elif self.__len__() < seed:
            raise ValueError("Select a smaller seed.")
        rows = n_examples
        fig, axes = plt.subplots(ncols=2, nrows=rows, figsize=(6, 3 * rows))

        for i in range(seed, seed + n_examples):
            hog_image = self.images[i]
            axes[i, 0].imshow(self.x[i])
            axes[i, 1].imshow(hog_image)
        plt.tight_layout()

        if savefig:
            plt.savefig("hogs_example.jpg", dpi=300)

        plt.show()

    def __getitem__(self, idx):
        """
        Returns the HOG feature vector for the given index.

        Parameters:
            idx (int): Index of the HOG feature vector.

        Returns:
            numpy.ndarray: HOG feature vector.
        """
        return self.values[idx]
    
    def __len__(self):
        """
        Returns the number of HOG feature vectors computed.

        Returns:
            int: Number of HOG feature vectors.
        """
        return len(self.hogs)


class ContourDescriptor:

    def __init__(self, x, n_points = 100, verbose = True):
        """
        ContourDescriptor class computes contour for a set of images.

        Parameters:
            x (aippa.utils.PotDataset or numpy.ndarray): Input images to compute contour descriptors. If x is of type
                "aippa.utils.PotDataset", it will be copied to self.x. If x is of type "numpy.ndarray" and has 2 dimensions,
                it will be assigned to self.x directly. Otherwise, self.x will be assigned as a numpy array of x.
            n_points (int): Number of points to interpolate on each contour (default: 100).
            verbose (bool): Flag to display progress bar (default: True).

        Attributes:
            x (numpy.ndarray): Input images to compute contour descriptors.
            contours (list): List of contour descriptors computed for each input image.
            contours_starting_rim (list): List of contours starting from rim computed for each input image.

        Returns:
            ContourDescriptor instance.

        Note:
            The contour descriptors are computed using the skimage.measure.find_contours function, for more information see: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours.
        """

        if type(x) == "aippa.utils.PotDataset":
            self.x = x[:]
        elif type(x) == "numpy.ndarray" and len(x.shape) == 2:
            self.x = x
        else:
            self.x = np.array(x)

        self.contours = []
        self.contours_starting_rim = []

        if verbose:
            for elem in tqdm(x, "Computing Contours"):
                tmp = measure.find_contours(elem, 0.5)[0]
                self.contours_starting_rim.append(tmp)
                tmp[:,0] = tmp[:,0].max() - tmp[:, 0]
                cnt = interpcurve(n_points, tmp[:,0], tmp[:,1])
                self.contours.append(cnt)
        else:
            for elem in x:
                tmp = measure.find_contours(elem, 0.5)[0]
                self.contours_starting_rim.append(tmp)
                tmp[:,0] = tmp[:,0].max() - tmp[:, 0]
                cnt = interpcurve(n_points, tmp[:,0], tmp[:,1])
                self.contours.append(cnt)



    def show_example(self, seed=0, n_examples=4, savefig = False):
        """
        Displays examples of the original images and their corresponding contours.

        Parameters:
            seed (int): Starting index of the examples to display (default: 0).
            n_examples (int): Number of examples to display (default: 4).
            savefig (bool): Flag to save the plot as an image file as "contours_example.jpg" (default: False).
        """
        assert seed < self.__len__(), "Seed to large, use a smaller seed"
        assert seed >= 0, "Can not use a seed < 0"
        if seed < self.__len__() < (seed + n_examples):
            n_examples = self.__len__() - seed
        elif self.__len__() < seed:
            raise ValueError("select a smaller seed")
        rows = n_examples
        fig, axes = plt.subplots(ncols=2, nrows=rows, figsize=(6,3*rows))
        if n_examples == 1:
            contour = self.__getitem__(0)
            axes[0].imshow(self.x[0] if self.typemy=="potds" else self.x)
            axes[1].scatter(*contour.T[::-1], s=4)
            axes[1].axis('equal')
        else:
            for i in range(seed, seed + n_examples):
                contour = self.__getitem__(i)
                axes[i, 0].imshow(self.x[i])
                axes[i, 1].scatter(*contour.T[::-1], s=4)
                axes[i, 1].axis('equal')
        plt.tight_layout()
        if savefig:
            plt.savefig("contours_example.jpg", dpi = 300)

        plt.show()

    def __getitem__(self, idx):
        return self.contours[idx]
    
    def __len__(self):
        return len(self.contours)

    



def Vectorization(x, verbose = True):
    """
    Extracts vector representations from input data.

    Parameters:
        x (aippa.utils.PotDataset or numpy.ndarray): Input data to extract vector representations from. If `x` is of type
            "aippa.utils.PotDataset", it will be copied to `x`. If `x` is of type "numpy.ndarray" and has 2 dimensions,
            it will be assigned to `x` directly. Otherwise, `x` will be assigned as a numpy array of `x`.
        verbose (bool): Flag to display progress bar (default: True).

    Returns:
        numpy.ndarray: Vector representations of the input data.

    """
    vect = []

    if type(x) == "aippa.utils.PotDataset":
        x = x[:]
    elif type(x) == "numpy.ndarray" and len(x.shape) == 2:
        x = x
    else:
        x = np.array(x)
    if verbose:
        for elem in tqdm(x, "Extracting vector"):
            vect.append(elem.flatten())
    else:
        for elem in x:
            vect.append(elem.flatten())

    vect = np.array(vect)

    return vect


    

class EllipticFourierAnalysis():
    
    def __init__(self, countours, order = 10, normalize=False, verbose = True):
        """
        EllipticFourierAnalysis class computes the elliptic Fourier descriptors for a set of contours.

        Parameters:
            countours (list): List of contours represented as a list of 2D points.
            order (int): Number of Fourier coefficients to compute (default: 10).
            normalize (bool): Flag indicating whether to normalize the Fourier coefficients (default: False).
            verbose (bool): Flag to display progress bar (default: True).

        Attributes:
            countours (list): List of contours represented as a list of 2D points.
            values (numpy.ndarray): Array of computed elliptic Fourier descriptors for each contour.
            visualize (numpy.ndarray): Array of computed elliptic Fourier descriptors for each contour, preserving the original shape.

        Returns:
            EllipticFourierAnalysis instance.

        Note:
            The elliptic Fourier descriptors are computed using the pyefd.elliptic_fourier_descriptors function (https://github.com/hbldh/pyefd).
            For more information about elliptic Fourier descriptors, see: https://en.wikipedia.org/wiki/Elliptic_Fourier_descriptors.

        """

        if countours[0].shape[1] != 2:
            raise ValueError("The countours must be a list of 2D points")
        
        self.countours = countours
        values = []
        visualize = []
        if verbose:
            for elem in tqdm(countours, "Defining elliptic Fourier descriptors"):
                coeffs = pyefd.elliptic_fourier_descriptors(elem, order=order, normalize=normalize)
                values.append(coeffs.ravel())
                visualize.append(coeffs)
        else:
            for elem in countours:
                coeffs = pyefd.elliptic_fourier_descriptors(elem, order=order, normalize=normalize)
                values.append(coeffs.ravel())
                visualize.append(coeffs)

        self.values = np.array(values)
        self.visualize = np.array(visualize)
    

    def harmonics_comparison(self, harmonic_numbers = [2, 5, 10, 15, 20, 30], normalize=False, figsize = (20,6), savefig = False, **kwargs):
        """
        Compare the reconstructed contours based on different harmonic numbers.

        Parameters:
            harmonic_numbers (list): List of harmonic numbers to compare (default: [2, 5, 10, 15, 20, 30]).
            normalize (bool): Flag indicating whether to normalize the Fourier coefficients (default: False).
            figsize (tuple): Figure size (width, height) in inches (default: (20, 6)).
            savefig (bool): Flag to save the plot as an image file as "harmonics_comparison.jpg" (default: False).
            **kwargs: Additional keyword arguments to pass to the `pyefd.reconstruct_contour` function.

        Returns:
            None

        Note:
            This method visualizes the comparison of reconstructed contours based on different harmonic numbers.
            It randomly selects a contour from the available contours and plots the reconstructed contours for each harmonic number.
            The elliptic Fourier descriptors are computed using the `pyefd.elliptic_fourier_descriptors` function, and the contours are reconstructed using the `pyefd.reconstruct_contour` function.
            For more information about elliptic Fourier descriptors and contour reconstruction, see: https://en.wikipedia.org/wiki/Elliptic_Fourier_descriptors.

        """

        fig, ax = plt.subplots(1, len(harmonic_numbers), figsize=figsize)
        fig.suptitle("Harmonics comparison", fontsize=15)

        selected_contour = self.countours[np.random.randint(0, self.countours.__len__())]

        for i, n in enumerate(harmonic_numbers):
            ax[i].set_title(f"{n} harmonics")
            armonic = pyefd.elliptic_fourier_descriptors(selected_contour, order=n, normalize=normalize)
            recostructed = pyefd.reconstruct_contour(armonic, locus=(0, 0), num_points=300, **kwargs)
            ax[i].plot(*recostructed.T[::-1])
            ax[i].axis('equal')

        if savefig:
            plt.savefig("harmonics_comparison.jpg", dpi = 300)

    plt.show() 


    def sum_efa(self, l):
        """
        Compute the sum of elements in a list.

        Parameters:
            l (list): List of numeric elements.

        Returns:
            float: The sum of elements in the list.

        """

        sum = 0
        for x in l:
            sum += x
        return sum


    def contour_comparison(self, info_selected, my_list, figsize = (5, 5), savefig = False, **kwargs):
        """
        Compare and plot contours based on selected information.

        Parameters:
            info_selected (pandas.DataFrame): DataFrame containing the selected information.
            my_list (pandas.Series): Series object representing the selected values to compare contours.
            figsize (tuple): Figure size (width, height) in inches (default: (5, 5)).
            savefig (bool): Flag to save the plot as an image file as "contour_comparison.jpg" (default: False).
            **kwargs: Additional keyword arguments to be passed to the plot function (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html).

        Returns:
            None
        """
        
        my_list_unique = my_list.dropna().unique()
        info_selected_resetted = info_selected.reset_index()
            
        rows = len(my_list_unique)

        fig, ax = plt.subplots(ncols=1, nrows=rows, figsize=(figsize[0],figsize[1]*rows))

        for i, nom in enumerate(my_list_unique):
            selected_rows = info_selected_resetted.loc[info_selected_resetted[my_list.name]== nom]
            selected_visualize = self.visualize[selected_rows.index]
###
            if rows == 1:
                for n, elem in enumerate(selected_visualize):
                    reconstructed_sel = pyefd.reconstruct_contour(elem, locus=(0, 0), num_points=300)
                    ax.plot(*reconstructed_sel.T[::-1], **kwargs)
                    ax.set_title(f"{nom}")
                    ax.axis('equal')
###
            else:
                for n, elem in enumerate(selected_visualize):
                    reconstructed_sel = pyefd.reconstruct_contour(elem, locus=(0, 0), num_points=300)
                    ax[i].plot(*reconstructed_sel.T[::-1], **kwargs)
                    ax[i].set_title(f"{nom}")
                    ax[i].axis('equal')

        if savefig:
            plt.savefig("contour_comparison.jpg", dpi = 300)


    def plot_average(self, info_selected, my_list, method = "mean", plot_all = False, alpha = 0.2, figsize = (5, 5), savefig = False):
        """
        Plot the average or median contour for each selected value.

        Parameters:
            info_selected (pandas.DataFrame): DataFrame containing the selected information.
            my_list (pandas.Series): Series object representing the selected values.
            method (str): Method for computing the average contour. Options: "mean" (default), "median".
            plot_all (bool): Flag indicating whether to plot all contours in addition to the average/median contour (default: False).
            alpha (float): Alpha value for the contours. Valid when plot_all = True (default: 0.2).
            figsize (tuple): Figure size (width, height) in inches (default: (5, 5)).
            savefig (bool):  Flag to save the plot as an image file as "plot_average.jpg" (default: False).

        Returns:
            None
        """

        my_list_unique = my_list.dropna().unique()
        info_selected_resetted = info_selected.reset_index()

        rows = len(my_list_unique)

        fig, ax = plt.subplots(ncols=1, nrows=rows, figsize=(figsize[0],figsize[1]*rows))      

        for i, nom in enumerate(my_list_unique):
            selected_rows = info_selected_resetted.loc[info_selected_resetted[my_list.name]== nom]
            selected_visualize = self.visualize[selected_rows.index]
            
            selected_visualize_list = list(selected_visualize)

            if method == "mean":
                prototype = pyefd.reconstruct_contour(np.mean((selected_visualize_list), axis=0), locus=(0, 0), num_points=300)
            elif method == "median":
                prototype = pyefd.reconstruct_contour(np.median((selected_visualize_list), axis=0), locus=(0, 0), num_points=300)
            else:
                raise ValueError("Select 'mean' or 'median' method")
            
            if rows == 1:
                ax.plot(*prototype.T[::-1], color = "black", linewidth = 2)
                ax.set_title(f"{nom}")
                ax.axis('equal')
            else:
                ax[i].plot(*prototype.T[::-1], color = "black", linewidth = 2)
                ax[i].set_title(f"{nom}")
                ax[i].axis('equal')

            if plot_all:
                if rows == 1:
                    for n, elem in enumerate(selected_visualize):
                        reconstructed_sel = pyefd.reconstruct_contour(elem, locus=(0, 0), num_points=300)
                        ax.plot(*reconstructed_sel.T[::-1], alpha = alpha, color = "grey")
                        ax.set_title(f"{nom}")
                        ax.axis('equal')
                else:
                    for n, elem in enumerate(selected_visualize):
                        reconstructed_sel = pyefd.reconstruct_contour(elem, locus=(0, 0), num_points=300)
                        ax[i].plot(*reconstructed_sel.T[::-1], alpha = alpha, color = "grey")
                        ax[i].set_title(f"{nom}")
                        ax[i].axis('equal')





        
        if savefig:
            plt.savefig("plot_average.jpg", dpi = 300)

    plt.show()

        
