# from time import time
# from PIL import Image
# import numpy as np



import numpy as np
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from time import time
from PIL import Image

import matplotlib.pyplot as plt
n_colors=64


oran = Image.open("k4.jpg")

oran = np.array(oran, dtype=np.float64) / 255

w,h,d=original_shape=tuple(oran.shape)
assert d==3
image_array = np.reshape(oran, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))


codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(oran)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()


#TODO HERE IS THE WORST ALGORITHM TO GET THE NUMBER OF COLORS IN IMAGE !!! I STILL WAITING MOR THAN  100SECONDS WITH 3.5GHZ WITHOUT ANY RESULT


image = Image.open("oran.jpg")
def buildArrray(image):
    image = np.asarray(image)
    array=[]
    nbc = len(image[0])
    nbl = len(image)
    i = 0
    j = 0
    while i < nbl:
        j=0
        while j < nbc:
            array.append(image[i][j])
            j = j + 1
        i = i + 1
    return array

def insertIfNotExist(colors, element):

    i=0
    while i< len(colors):
        if (colors[i] == element).all():
            return colors
        i=i+1
    colors.append(element)
    return colors


def NumberOfColors(image):
    image=buildArrray(image)
    colors=[]

    i=0
    while i<len(image):
        colors = insertIfNotExist(colors, image[i])
        i=i+1
    return len(colors)

#print NumberOfColors(image)

