import os
import re
from fnmatch import fnmatch 
import numpy as np
import statistics
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
                image_names.append(name)
                images.append(read_image(os.path.join(root, name)))
    images = np.array(images)
    return image_names, images

def read_image(path):
    img = Image.open(path)
    img = np.array(img)
    img = resize(img,(288,288, 3), anti_aliasing=True)
    img = img - 0.5
    return img

def test_single():
    original = read_image("ImageNet_results/black_results_imagenet_rgb/imagenet/39_original_id38_differ10_True_l211.45_pdistance0.004899389576166868.png")
    original = np.array([original])
    adversarial = read_image("ImageNet_results/black_results_imagenet_rgb/imagenet/39_adversarial_id38_differ10_True_l211.45_pdistance0.004899389576166868.png")
    adversarial = np.array(adversarial)
    hash_model = ImageNet_HashModel(10, hash_length, 4)
    res = hash_model.compute_hash_distance(original, adversarial)


'''
Perform attack. Count the average false positive rate.
'''
hash_length = 8
def test_performance(attack_image_name, attack_images, original_image_name, original_images, threshold):
    '''
    @param
    attack_images, original_images: numpy array
    '''
    hash_model = ImageNet_HashModel(threshold, hash_length, 4)
    false_positive = []
    num_top_5 = 0
    num_top_1 = 0
    for i, attack_image in enumerate(attack_images):
        res = hash_model.compute_hash_distance(original_images, attack_image)
        false_positive.append(false_positive_rate(res, threshold, original_image_name, attack_image_name[i]))
        if (top_k_hit(res, threshold, original_image_name, attack_image_name[i], 5)):
            num_top_5 += 1
        if (top_k_hit(res, threshold, original_image_name, attack_image_name[i], 1)):
            num_top_1 += 1
    avg = statistics.mean(false_positive)
    return num_top_1 / len(attack_image_name), num_top_5 / len(attack_image_name), avg


def top_k_hit(result, threshold, original_image_names, attack_image_name, k):
    '''
    Whether hit within k images.
    '''
    index_result = []
    for i in range(len(result)):
        index_result.append([result[i],i])
        
    reverse_res = []
    for i in range(len(index_result)):
        if (index_result[i][0] < threshold):
            reverse_res.append([index_result[i][0], index_result[i][1]])
    ''' sort'''
    reverse_res.sort(key=lambda x:x[0])
    if (reverse_res == []):
        print("error")
        return False
    for i in range(k):
        if (original_image_names[reverse_res[i][1]][:2] == attack_image_name[:2]):
            return True

    return False


def false_positive_rate(result, threshold, original_image_name, attack_image_name):
    '''
    Compute the false positive rate.
    '''
    false_positive = 0
    for i in range(len(result)):
        if result[i] < threshold:
            print(original_image_name[i], attack_image_name)
            if original_image_name[i][:2] != attack_image_name[:2]:
                false_positive += 1
    return false_positive
            

if __name__ == '__main__':     
    attack_image_name, attack_images = load_image('./ImageNet_results/black_results_imagenet_rgb/imagenet/', r".*adversarial.*png$")
    original_image_name, original_images = load_image('./ImageNet_results/black_results_imagenet_rgb/imagenet/', r".*original.*png$")
    top1, top5, average_false_positve_rate = test_performance(attack_image_name, attack_images, original_image_name, original_images, 0.4)
    print(top1, top5, average_false_positve_rate)