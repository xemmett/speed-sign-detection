"""
Kacper Dudek - 18228798
Christian Ryan - 18257356
Charlie Gorey O'Neill - 18222803
Sean McTiernan - 18224385
"""


from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
# mpl.rc('image', cmap='gray')

from skimage.color import rgb2hsv
from skimage.io import imread, imshow
from skimage.morphology import binary_dilation
from skimage.measure import regionprops
from skimage.transform import resize

from scipy.ndimage import generate_binary_structure, label

def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')

class RegionProposer():

    def __init__(self, filename, verbose: bool = 0) -> None:
        self.filename = filename

        # load img
        self.img = self.load_image(filename)
        # self.img[:,:,0] = np.floor(self.img[:,:,0] * 1.05).astype(np.uint8)
        # plot_comparison(self.img, adjusted_gamma_image, 'contrast boost')

        # hue, saturation, value handling
        hue, saturation, value = self.rgb_to_hsv(self.img, verbose=verbose)

        h = self.threshold_channel(hue, threshold=0.8, verbose=verbose, operator='>=', channel_name='hue')
        h = self.threshold_channel(hue, threshold=0.05, verbose=verbose, operator='<=', channel_name='hue')

        s = self.threshold_channel(saturation, threshold=0.25, verbose=verbose, operator='>', channel_name='saturation')
        v = self.threshold_channel(value, threshold=0.125, verbose=verbose, operator='>', channel_name='value')

        mask = h * s * v
        filtered_img = self.filtering(mask, verbose)
        # self.show_image(filtered_img)

        # region labelling
        self.regions = self.region_labelling(filtered_img, self.img, verbose)
        
        if(verbose): plt.show()

    def load_image(self, fp: str ='') -> Image:
        image = imread(fp)[:,:,:3] # open image, since is png, 4 channels are proposed, we just take first 3
        self.current_image = image
        return image
    
    def rgb_to_hsv(self, img: Image, verbose: bool = 0):

        hsv_image = rgb2hsv(img)
        hue = hsv_image[:, :, 0]
        saturation = hsv_image[:, :, 1]
        value = hsv_image[:, :, 2]

        if(verbose):
            
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(8, 3))

            ax0.imshow(img)
            ax0.set_title("RGB image")
            ax0.axis('off')
            ax0.axis('off')
            ax1.imshow(hue, cmap='hsv')
            ax1.set_title("Hue channel")
            ax1.axis('off')
            ax2.imshow(value)
            ax2.set_title("Value channel")
            ax2.axis('off')
            ax3.imshow(saturation)
            ax3.set_title("Saturation channel")
            ax3.axis('off')

            fig.tight_layout()

        return hue, saturation, value

    def threshold_channel(self, hsv_channel, threshold: float = 0.04, verbose: bool = 0, operator: str = '>', channel_name=None):
        if(operator == '>'):
            binary_img = hsv_channel > threshold
        elif(operator == '<'):
            binary_img = hsv_channel > threshold
        elif(operator == '>='):
            binary_img = hsv_channel >= threshold
        elif(operator == '<='):
            binary_img = hsv_channel <= threshold
        else:
            return print('no compatible operator: >, <, >=, <=')

        if(verbose):
            fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 3))

            ax0.hist(hsv_channel.ravel(), 512)
            ax0.set_title("Histogram of the {} channel with threshold".format(channel_name))
            ax0.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2)
            ax0.set_xbound(0, 1)
            ax1.imshow(binary_img)
            ax1.set_title("thresholded image")
            ax1.axis('off')

            fig.tight_layout()

        return binary_img

    def filtering(self, img_mask, verbose: bool = 0):
        
        filtered_img = binary_dilation(img_mask) # fill in gaps in images.
        for i in range(5):
            filtered_img = binary_dilation(filtered_img) # fill in gaps in images.

        if(verbose):
            plot_comparison(img_mask, filtered_img, 'filtered')

        return filtered_img

    def show_image(self, img: Image):
        imshow(img)
    
    def region_labelling(self, img, original_img, verbose: bool = 0):
        regions = []
        s = generate_binary_structure(2,2)
        labelled_array, num_features = label(img)
        
        if(verbose):
            print(num_features)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(original_img)

        for i, region in enumerate(regionprops(labelled_array)):
            
            if(verbose): print("labeled_region[{}].area = {}".format(i, region.area))

            if(region.area >= 441):
                # draw rectangle
                x, y, width, height = region.bbox
                # cropped_img = self.crop_image(x, y, width, height)
                # test for aspect ratio
                # print(i, width / height)
                # print(i, height)
                # print(i, width)
                # if(((height / width) < 2 and (width / height) > 0.6) or ((width / height) < 2 and (height / width) > 0.6)):
                region = self.crop_image(self.img, x, y, width, height)
                regions.append(region)

                if(verbose):
                    rect = Rectangle((y, x), height - y, width - x, fill=0, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
                    plt.text(y, x, f"[{i}] {height / width}", color="orange", fontdict={"fontsize":20})

        return regions
    
    def crop_image(self, img, x, y, width, height):
        return img[x:width, y:height]

# from os import listdir

# for filename in listdir('speed-sign-test-images'):
#     if(filename.endswith('.png')):
#         # set verbose to 0 for no graphs or anything
#         RegionProposer(f"speed-sign-test-images/{filename}", verbose=0).regions

        