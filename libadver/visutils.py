import numpy as np
from PIL import Image
from PIL import ImageDraw,ImageFont
import PIL
import copy
import matplotlib.cm as mpl_color_map

## visualize function
def recreate_image(im_as_var, mean, std):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate [3 * 224 * 224]
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    #reverse_mean = [-0.485, -0.456, -0.406]
    #reverse_std = [1/0.229, 1/0.224, 1/0.225]
    reverse_mean = [-i for i in mean]
    reverse_std = [1/i for i in std]
    recreated_im = copy.copy(im_as_var.data.numpy())
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im
def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """

    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = file_name
    save_image(gradient, path_to_file)
def save_image(im, path):
    """
        Saves a numpy matrix of shape D(1 or 3) x W x H as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image

    TODO: Streamline image saving, it is ugly.
    """
    if isinstance(im, np.ndarray):
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=0)
            print('A')
            print(im.shape)
        if im.shape[0] == 1:
            # Converting an image with depth = 1 to depth = 3, repeating the same values
            # For some reason PIL complains when I want to save channel image as jpg without
            # additional format in the .save()
            print('B')
            im = np.repeat(im, 3, axis=0)
            print(im.shape)
            # Convert to values to range 1-255 and W,H, D
        # A bandaid fix to an issue with gradcam
        if im.shape[0] == 3 and np.max(im) == 1:
            im = im.transpose(1, 2, 0) * 255
        elif im.shape[0] == 3 and np.max(im) > 1:
            im = im.transpose(1, 2, 0)
        im = Image.fromarray(im.astype(np.uint8))
    im.save(path)

def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'jet')
    # Save colored heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.png')
    print(np.max(heatmap))
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    print()
    print(np.max(heatmap_on_image))
    path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    print()
    print(np.max(activation_map))
    path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)

def generate_colormap(activation, colormap_name):
    """
        Generate heatmap on image (PIL.Image)
    Args:
        activation_map (numpy arr): Activation map (grayscale) 0~255 shape [224,224]
        colormap_name (str): Name of color map
    """
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))
    return no_trans_heatmap

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


## Draw function
def draw_text(im, text, color, length):
    """
        Draw text on Image
    Args:
        im (PIL image): Original image
        text (str): [Proliferative DR: 98.2%]
        color (str): [Green]
        length (int): length of background for text
    Return:
        im (PIL Image)
    """

    if not Image.isImageType(im):
        raise TypeError("Input image should be the instance of PIL.Image")


    font = ImageFont.truetype("arial.ttf",20)
    # im.paste(til1,(0,224-30))
    til2 = Image.new("RGB",(length,30),color)
    im.paste(til2,(0,224-30))
    draw = ImageDraw.Draw(im)
    draw.text((10,198), text,"white",font=font)
    return im

if __name__=="__main__":
    ## test Draw function
    color1 = "green"
    color2 = "red"
    #
    im1 = Image.open('original_5.png')
    text1 = "Proliferative DR: 98.2%"
    im1 = draw_text(im1,text1,color1,length=224)
    #im.save("pro_reconstructed5.tiff")
    im1.save("pro_original5.tiff")
    im1.save("pro_original5.png")
