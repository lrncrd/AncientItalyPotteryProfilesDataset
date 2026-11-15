import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage import transform
from scipy import sparse
import pandas as pd
import geopandas
from skimage.filters import median


def cylinder_vol(r, h):
    return np.pi * h * (r)**2
def cylinder_sup(r, h):
    return 2*(np.pi) * r * h
def cm2tom2(x):
    return x/10000
def cm3tol(x):
    return x/1000



def read_img_files(base_dir, img_extensions=['.jpg', '.jpeg', '.png']):
    """
    Read image files from a directory.

    Parameters:
        base_dir (str or pathlib.Path): Base directory path.
        img_extensions (list): List of image file extensions to consider. Default is ['.jpg', '.jpeg', '.png'].

    Returns:
        list: List of image file paths.
    """
    base_dir = Path(base_dir)
    img_files = []
    for elem in os.listdir(base_dir):
        if os.path.isdir(base_dir /elem):
            for x in os.listdir(base_dir / elem):
                if (base_dir / elem / x).suffix in img_extensions:
                    img_files.append(base_dir / elem / x)
        elif  (base_dir / elem).suffix in img_extensions:
            img_files.append(base_dir / elem)
        else:
            print(f"something strange happened for file {elem}")
    img_files.sort()
    return img_files


def pad_image_fixed_dim_rescaled(img, dim):
    """
    Pad an image to a fixed dimension by rescaling and centering the image.

    Parameters:
        img (numpy.ndarray): Input image as a numpy array.
        dim (tuple): Desired dimensions (height, width) for the padded image.

    Returns:
        numpy.ndarray: Padded image.

    """
    img_shape = img.shape
    dh = dim[0] - img_shape[0]
    dw = dim[1] - img_shape[1]


    if not dh % 2:
        a = (dh//2, dh//2)
    else:
        a = (dh//2, dh//2 + 1)

    if not dw % 2:
        b = (dw//2, dw//2)
    else:
        b = (dw//2, dw//2 + 1)

    padded = np.pad(img, (a, b))
    return padded


def pad_image_fixed_dim_padded(img, dim):
    """
    Pad an image to a fixed dimension by adding padding around the image.

    Parameters:
        img (numpy.ndarray): Input image as a numpy array.
        dim (int): Desired dimension for the padded image (height = width).

    Returns:
        numpy.ndarray: Padded image.

    """
    img_shape = img.shape
    dh = dim - img_shape[0]
    dw = dim - img_shape[1]

    if not dh % 2:
        a = (dh//2, dh//2)
    else:
        a = (dh//2, dh//2 + 1)

    if not dw % 2:
        b = (dw//2, dw//2)
    else:
        b = (dw//2, dw//2 + 1)

    padded = np.pad(img, (a, b))
    return padded


def minimum_image(img, margin=1):
    """
    Crop the minimum bounding box around a non-zero region in an image.

    Parameters:
        img (numpy.ndarray): Input image as a numpy array.
        margin (int): Margin to add around the minimum bounding box.

    Returns:
        numpy.ndarray: Minimum image.

    """
    c_hull = np.where(img > 0)
    x_min, x_max = c_hull[0].min(), c_hull[0].max()
    y_min, y_max = c_hull[1].min(), c_hull[1].max()
    min_img = img[x_min:x_max,y_min:y_max]
    min_img = np.pad(min_img, margin)

    return min_img


def denoise_and_binarize(img):
    """
    Denoise an image using morphological operations and then binarize it using Otsu's thresholding.

    Parameters:
        img (numpy.ndarray): Input image as a numpy array.

    Returns:
        numpy.ndarray: Binarized image.

    """
    selem = morphology.disk(3)
    x = img > threshold_otsu(img)
    x = morphology.closing(x, selem)

    return x


def pot_metrics(img, pixel_cm):
    """
    Calculate metrics related to a pot shape represented by a binary image.

    Parameters:
        img (numpy.ndarray): Binary image of the pot shape.
        pixel_cm (float): Conversion factor from pixels to centimeters.

    Returns:
        tuple: A tuple containing the volume, maximum radius, and surface area of the pot shape.


    Description:
        The `pot_metrics` function calculates various metrics related to a pot shape based on a binary image representation
        of the pot. The function takes in the binary image `img` and the conversion factor `pixel_cm` to convert pixel
        measurements to centimeters.

        The function calculates the following metrics:
        - Volume: The volume of the pot shape, representing the space enclosed by the pot. The volume is calculated by summing
        the volumes of individual vertical cylindrical slices of the pot shape.
        - Maximum Radius: The maximum radius of the pot shape, representing the largest distance from the center to the
        outer edge of the pot. The maximum radius is obtained from the farthest point on the pot shape.
        - Surface Area: The surface area of the pot shape, representing the external area of the pot. The surface area is
        calculated by summing the areas of the upper parts of the pot, assuming the pot shape extends infinitely upwards.
    """
    rows, cols = img.shape
    volume = 0
    surface = 0
    radii = []
    for i in range(rows):
        row = img[i, :]
        if sum(row) > 0:
            real_row = row
            real_row_min = np.min(np.where(real_row == True))
            radius_sup = cols - real_row_min
            radius_sup_cm = radius_sup * pixel_cm

            surface += cylinder_sup(radius_sup_cm, pixel_cm)      
        else:
            pass

    for i in range(rows):
        row = img[i, :]
        if sum(row) > 0:
            real_row = row
            real_row_max = np.max(np.where(real_row == True))
            radius_vol = cols - real_row_max
            if radius_vol < 2:
                break
            radius_vol_cm = radius_vol * pixel_cm
            radii.append(radius_vol)

            volume += cylinder_vol(radius_vol_cm, pixel_cm)    
        else:
            pass

    return volume, radii[0], surface


def image_dataset_standardize(base_dir, diagnostic=False, pad_to_fit=True, rel_size=128, pixel_cm = 0.125, margin=1):
    """
    Standardize an image dataset by applying preprocessing steps.

    Parameters:
        base_dir (str): Directory path containing the image files.
        diagnostic (bool): Flag indicating whether to run the function in diagnostic mode (process a subset of images).
            Default is False.
        pad_to_fit (bool): Flag indicating whether to pad images to fit the largest image size in the dataset. If False,
            the images will be rescaled and uniformly padded to a relative size specified by `rel_size`. Default is True.
        rel_size (int): Relative size for rescaling and padding the images. Used only if `pad_to_fit` is False.
            Default is 128.
        pixel_cm (float): Conversion factor from pixels to centimeters. Default is 0.125.
        margin (int): Margin value for padding the images. Default is 1.

    Returns:
        tuple: A tuple containing the standardized images, image file paths, sizes in pixels, volumes, diameters, and
        surfaces of the pots.
    
    Description:
        The `image_dataset_standardize` function reads and preprocesses a dataset of pot images stored in the specified
        `base_dir`. It applies a series of preprocessing steps to standardize the images for further analysis.

        The function performs the following steps:
        1. Reads the image files from the specified directory using the `read_img_files` function.
        2. If `diagnostic` mode is enabled, selects a subset of images for processing.
        3. Reads each image, converts it to grayscale, and applies denoising and binarization using the
        `denoise_and_binarize` function.
        4. Removes unnecessary background space from the image using the `minimum_image` function.
        5. Calculates metrics such as volume, diameter, and surface area of the pot shape using the `pot_metrics` function.
        6. Rescales or pads the images based on the `pad_to_fit` flag and `rel_size` value.
        7. Returns the standardized images, image file paths, sizes in pixels, volumes, diameters, and surfaces of the pots.
    """
    img_files = read_img_files(base_dir)
    assert len(img_files) > 0, f"Error: no image files found in the specified path({base_dir})"

    if diagnostic:
        img_files = img_files[:100]

    images = []
    sizes_px = []
    volumes = []
    diameters = []
    surfaces = []
    

    for img_file in tqdm(img_files, f"Reading images from disk and cleaning"):
        x = Image.open(img_file).convert("L")
        x = np.array(x)
        x = 255 - x
        x = denoise_and_binarize(x)
        x = minimum_image(x)
        images.append(x)
        sizes_px.append(x.shape)
        vol, radius, surface = pot_metrics(x, pixel_cm)
        
        volumes.append(np.round(cm3tol(vol), 3))
        diameters.append(2*radius*pixel_cm)
        surfaces.append(np.round(cm2tom2(surface), 5))
        
    max_size = np.array(sizes_px).max()

        

    images_refined = []
    
    if pad_to_fit:
        for img in tqdm(images, f"Padding images to standard size: {max_size}"):
            padded = pad_image_fixed_dim_padded(img, max_size)
            images_refined.append(padded)
    else:
        for img in tqdm(images, f"Rescaling and unfiforming images to size: {rel_size}"):
            rescaled_img = transform.rescale(img, (rel_size - margin) / max(img.shape) )
            padded = pad_image_fixed_dim_rescaled(rescaled_img, np.array((rel_size, rel_size)))
            images_refined.append(padded)

    return images_refined, img_files, sizes_px, volumes, diameters, surfaces


def make_dataset(base_dir, out_path, diagnostic=False, pad_to_fit=True, rel_size=128, pixel_cm = 0.125, get_metrics = True):
    """
    Create a standardized dataset from a directory of pot images.

    Parameters:
        base_dir (str): Directory path containing the image files.
        out_path (str): Directory path to save the dataset file.
        diagnostic (bool): Flag indicating whether to run the function in diagnostic mode (process a subset of images) (default: False).
        pad_to_fit (bool): Flag indicating whether to pad images to fit the largest image size in the dataset. If False, the images will be rescaled and uniformly padded to a relative size specified by rel_size. (default: True)
        rel_size (int): Relative size for rescaling and padding the images. Used only if pad_to_fit is False (default: 128)
        pixel_cm (float): Conversion factor from pixels to centimeters (default: 0.125)
        get_metrics (bool): Flag indicating whether to calculate and save pot metrics in the dataset file (default: True)

    Description:
        The `make_dataset` function creates a standardized dataset from a directory of pot images stored in the specified
        `base_dir`. It applies preprocessing steps and saves the standardized dataset in a specified output directory.

        The function performs the following steps:
        1. Creates the output directory if it doesn't exist.
        2. Calls the `image_dataset_standardize` function to obtain standardized images and pot metrics.
        3. Reshapes the images to use CSR matrices.
        4. Saves the images as a CSR matrix in a compressed NPZ file in the output directory.
        5. If `get_metrics` is True, saves the pot metrics in a CSV file in the output directory.
        6. If `get_metrics` is False, saves only the image IDs in a CSV file in the output directory.
        7. Prints a completion message indicating the location of the dataset file.
    """
    out_path = Path(out_path).resolve()
    os.makedirs(out_path, exist_ok=True)

    images_refined, img_files, sizes, volumes, diameters, surfaces  = image_dataset_standardize(base_dir, diagnostic=diagnostic, pad_to_fit=pad_to_fit, rel_size=rel_size, pixel_cm = pixel_cm)
    # reshape to use csr matrices
    images_refined = np.array(images_refined).reshape((len(images_refined), -1))
    images_sparse = sparse.csr_matrix(images_refined)

    sparse.save_npz(out_path / f"images_csr_{'padded' if pad_to_fit else 'rescaled'}_{int(np.sqrt(images_refined.shape[1]))}.npz", images_sparse)
    if get_metrics:
        with open(out_path / "info.csv", "w", encoding="utf_8") as f:
            f.write('ids,height (cm),width (cm),volume (l),diameter (cm),surface (mq)\n')
            for i, name in enumerate(img_files):
                f.write(f"{name.stem},{sizes[i][0]*pixel_cm},{sizes[i][1]*pixel_cm*2},{volumes[i]},{diameters[i]},{surfaces[i]}\n")
    else:
        with open(out_path / "info.csv", "w", encoding="utf_8") as f:
            f.write('ids\n')
            for i, name in enumerate(img_files):
                f.write(f"{name.stem}\n")
        

    print(f"Completed!\nDataset file created at {out_path}")




class PotDataset:

    def __init__(self, ds_path, archeodata_path=None, geo_path =None):
        """
        A class representing a dataset of pot images.

        Parameters:
            ds_path (str): Directory path of the dataset realised with the make_dataset function.
            archeodata_path (str or list): Path(s) to the archeological data file(s) (default: None).
            geo_path (str or list): Path(s) to the geographical data file(s) (default: None).
        
        Returns:
            PotDataset instance.
        """
        ds_path = Path(ds_path)
        #if archeodata_path:
        if archeodata_path is not None:
            archeodata = pd.read_excel(archeodata_path)
        #if geo_path:
        if geo_path is not None:
            self.geo = geopandas.read_file(geo_path)
        info = pd.read_csv(ds_path / 'info.csv')
        imgs_opts = [x for x in os.listdir(ds_path) if (ds_path/x).suffix == ".npz"]
        opts = [f'{i}: {x}' for i, x in enumerate(imgs_opts)]
        selected_df = int(0)#int(input("Select image dataset file:\n" + '\n'.join(opts) ))
        self.images = sparse.load_npz(ds_path / imgs_opts[selected_df])
        if archeodata_path:
            self.info = info.merge(archeodata, on="ids")
        else:
            self.info = pd.read_csv(ds_path / 'info.csv')
        self.img_size = np.sqrt(self.images.shape[1]).astype(np.int32)

   

    def show_example(self, seed=0, n_examples=4):
        """
        Display examples of pot images from the dataset.

        Parameters:
            seed (int): Seed index to start displaying the examples (default: 0)
            n_examples (int): Number of examples to display (default: 4)
        
        Returns:
            None
        
        Description:
            The `show_example` method displays a grid of pot image examples from the dataset, starting from the specified
            `seed` index and displaying `n_examples` images.

        """
        assert seed < self.__len__() - n_examples, "Seed to large, use a smaller seed"
        rows = int((n_examples + 1) / 2)
        fig, axes = plt.subplots(ncols=2, nrows=rows, figsize=(6,3*rows))
        axes = axes.ravel()
        for i in range(seed, seed + n_examples):
            img = self.__getitem__(i) 
            info = self.info.loc[i]
            # img = img.A
            axes[i - seed].imshow(img)
            axes[i - seed ].set_title(info["ids"])
        plt.tight_layout()
        plt.show()
    
    def by_name(self, name):
        """
        Displays the image of a pot in the dataset by searching for its name.

        Parameters:
        - name (str): The name of the pot to search for in the dataset.

        Returns:
        - None

        Raises:
        - AssertionError: If no pot with the specified name is found.
        - AssertionError: If multiple pots with the specified name are found.
        """
        idxs = np.where(self.info["ids"] == name)[0]
        assert len(idxs) > 0, f"No element found with name: {name}"
        assert len(idxs) == 1, f"Multiple element found with name: {name}"
        return plt.imshow(self.__getitem__(idxs[0]))
    
    def plot_items(self, n_examples = 10, figsize = (10, 10)):
        """
        Plots a grid of pot images from the dataset.

        Parameters:
        - n_examples (int): The number of pot images to display in both the rows and columns of the grid (default: 10).
        - figsize (tuple): Figure size (width, height) in inches (default: (10, 10))

        Returns:
        - None
        """
        fig, ax = plt.subplots(n_examples, n_examples, figsize=figsize,
                           subplot_kw=dict(xticks=[], yticks=[]))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        for i, axi in enumerate(ax.flat):
            im = axi.imshow(self[i], cmap='viridis')


    def __getitem__(self, idx):
        return self.images[idx].toarray().reshape((-1, self.img_size, self.img_size)).squeeze()


    def __len__(self):
        return self.images.shape[0]
    
