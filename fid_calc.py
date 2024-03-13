import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
import os
from PIL import Image
from scipy import linalg
 
# scale an array of images to a new size
def scale_images(images, new_shape):
 images_list = list()
 for image in images:
 # resize with nearest neighbor interpolation
    new_image = resize(image, new_shape, 0)
    # store
    images_list.append(new_image)
 return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
 # calculate activations
 act1 = model.predict(images1)
 act2 = model.predict(images2)
 print(act1.shape)   
 # calculate mean and covariance statistics

 mu1, sigma1 = act1.mean(axis=0), cov(act1.T , rowvar=False)
 mu2, sigma2 = act2.mean(axis=0), cov(act2.T , rowvar=False)
 
 mu1 = numpy.atleast_1d(mu1)
 mu2 = numpy.atleast_1d(mu2)

 sigma1 = numpy.atleast_2d(sigma1)
 sigma2 = numpy.atleast_2d(sigma2)
 #sigma1[sigma1 < 0] = 0
 #sigma2[sigma2 < 0] = 0
 temp = sigma1.dot(sigma2)
 #temp[temp < 0] = 0
 covmean, _ = linalg.sqrtm( temp, disp=False)
 # calculate sum squared difference between means
 
 if numpy.iscomplexobj(covmean):
    if not numpy.allclose(numpy.diagonal(covmean).imag, 0, atol=1e-3):
        m = numpy.max(numpy.abs(covmean.imag))
        covmean = covmean.real
    

 diff = mu1 - mu2  
 
 tr_covmean = numpy.trace(covmean)
 return (diff.dot(diff) + numpy.trace(sigma1)
            + numpy.trace(sigma2) - 2 * tr_covmean)
 
# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
# define two fake collections of images
dataset_path_1 = "/home/yasamin/Documents/Data_breast/Original/train/benign"
dataset_path_2 = "/home/yasamin/Documents/Data_breast/lora_2/Augmented_1.0_percent/no_adjective/train/benign"
images1_paths = [os.path.join(dataset_path_1, filename) for filename in os.listdir(dataset_path_1)]
images2_paths = [os.path.join(dataset_path_2, filename) for filename in os.listdir(dataset_path_2)]

# Load and preprocess images

images1 = []
images2 = []
for path in images1_paths:
    img = Image.open(path)
    img = img.resize((299, 299))  # Resize to 299x299
    img = numpy.array(img)
    images1.append(img)
images1 = numpy.array(images1)
for path in images2_paths:
    img = Image.open(path)
    img = img.resize((299, 299))  # Resize to 299x299
    img = numpy.array(img)
    images2.append(img)
images2 = numpy.array(images2)    

print('Prepared', images1.shape, images2.shape)

# resize images
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled', images1.shape, images2.shape)
# pre-process images
images1 = preprocess_input(images1/255)
images2 = preprocess_input(images2/255)
# fid between images1 and images1
#fid = calculate_fid(model, images1, images1)
#print('FID (same): %.3f' % fid)
# fid between images1 and images2

fid = calculate_fid(model, images1, images2)
print('FID (different): %.3f' % fid)


