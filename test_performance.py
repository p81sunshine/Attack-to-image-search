import os
import re
from fnmatch import fnmatch 
import numpy as np
from PIL import Image
from skimage.transform import rescale, resize, downscale_local_mean
from setup_imagenet_hash import ImageNet_HashModel

def load_image(folder, pattern):
    '''
    Returns a list of images and a list of corresponding names.
    '''
    print('loading images')
    images = []
    image_names = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if re.match(pattern, name):
                image_names.append(os.path.join(root, name))
                images.append(read_image(os.path.join(root, name)))
    images = np.array(images)
    return image_names, images

def read_image(path):
    img = Image.open(path)
    img = np.array(img)
    img = resize(img,(288,288, 3), anti_aliasing=True)
    img = img - 0.5
    return img

'''
Perform attack. Count the average false positive rate.
'''
hash_length = 8
def test_performance(attack_image_name, attack_images, original_image_name, original_images, threshold):
    hash_model = ImageNet_HashModel(threshold, hash_length, 4)
    res = hash_model.compute_hash_distance(original_images, original_images[0])
    print(res)

if __name__ == '__main__':     
    attack_image_name, attack_images = load_image('./ImageNet_results/black_results_imagenet_rgb/imagenet/', r".*adversarial.*png$")
    original_image_name, original_images = load_image('./ImageNet_results/black_results_imagenet_rgb/imagenet/', r".*original.*png$")
    test_performance(attack_image_name, attack_images, original_image_name, original_images, 0.2)