import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from scipy import ndimage, misc, sparse
import random
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import alphashape
import os

# %%
def get_points(N, dist, max_points):
    points = []
    x0 = np.random.rand()
    y0 = np.random.rand()
    z0 = np.random.rand()
    points.append([x0, y0, z0])
    while True:
        x = np.random.rand()
        y = np.random.rand()
        z = np.random.rand()
        if np.sqrt((x0-x)**2 + (y0-y)**2 + (z0-z)**2) <= dist:
            points.append([x, y, z])
            if len(points) == max_points:
                break
    return points 


def create_masks(N, grid, max_points, dist, it):

    #x = np.linspace(0, 1, N)
    #X,Y,Z = np.meshgrid(x,x,x)

    points = get_points(N, dist, max_points)    

    #alpha = 0.95*alphashape.optimizealpha(points)
    alpha = 30
    alpha_shape = alphashape.alphashape(points, alpha)

    mask = np.zeros([N*N*N])
    indices = alpha_shape.contains(grid)
    if sum(indices) == 0:
        create_masks(N, grid, max_points, dist, it)
        return

    mask[indices] = 1
    mask = mask.reshape([N, N, N])
    
    s1 = np.squeeze(np.sum(mask, 0))
    s2 = np.squeeze(np.sum(mask, 1))
    s3 = np.squeeze(np.sum(mask, 2))

    s1[s1!=0] = 1
    s2[s2!=0] = 1
    s3[s3!=0] = 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(s1)
    ax2.imshow(s2)
    ax3.imshow(s3)

    name = 'mask_' + str(it)
    np.save('masks/arrays/'+ name, mask.astype(np.uint8))
    plt.savefig('masks/plots/' + name)
    plt.close()

    #return mask

N = 256
max_points = 250
num_masks = 500

max_size = 0.15
min_size = 0.05

x = np.linspace(0, 1, N)
X, Y, Z = np.meshgrid(x,x,x) 
grid = [tuple(x) for x in np.array([X.flatten(), Y.flatten(), Z.flatten()]).T]
for it in range(num_masks):
    distrand = np.random.rand()
    dist = distrand*(max_size-min_size) + min_size
    start = time.time()
    mask = create_masks(N, grid, max_points, dist, it)
    end = time.time()
    print(end - start)
    print(str(it) + '/' + str(num_masks))
