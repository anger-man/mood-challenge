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

# %%
def get_points(N, dist, max_points):
    pointlist = []
    point0x = np.random.rand()
    point0y = np.random.rand()
    point0z = np.random.rand()
    pointlist.append([point0x, point0y, point0z])
    while True:
        pointx = np.random.rand()
        pointy = np.random.rand()
        pointz = np.random.rand()
        if np.sqrt((point0x-pointx)**2 + (point0y-pointy)**2 + (point0z-pointz)**2) <= dist:
            pointlist.append([pointx, pointy, pointz])
            if len(pointlist) == max_points:
                break
    return pointlist


def create_masks(N, grid, max_points, dist, it):

    x = np.linspace(0, 1, N)
    X,Y,Z = np.meshgrid(x,x,x)

    points = get_points(N, dist, max_points)

    alpha_shape = alphashape.alphashape(points, 2.0)

    mask = np.zeros([N*N*N])
    start = time.time()
    #idd = 0
    #for k in np.array([X.flatten(), Y.flatten(), Z.flatten()]).T:
    #    tmp_point = [(k[0],k[1],k[2])]
    #    if alpha_shape.contains(tmp_point):
    #        #xid.append(m)
    #        #yid.append(n)
    #        #zid.append(l)
    #    #if contained([m, n, l]):
    #        mask[idd] = 1
    #        idd += 1
    #    print('Completed ' + str(round(steps/total*100)) + '%')
    #    steps += 1
    #mask = mask.reshape([N, N, N])
    indices = alpha_shape.contains(grid)
    if sum(indices) == 0:
        create_masks(N, grid, max_points, dist, it)
        return

    end = time.time()
    print(end - start)

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
max_points = 40
num_masks = 10

max_size = 0.15
min_size = 0.05

x = np.linspace(0, 1, N)
X, Y, Z = np.meshgrid(x,x,x) 
grid = [tuple(x) for x in np.array([X.flatten(), Y.flatten(), Z.flatten()]).T]
for it in range(num_masks):
    distrand = np.random.rand()
    dist = distrand*(max_size-min_size) + min_size
    mask = create_masks(N, grid, max_points, dist, it)
    print(str(it) + '/' + str(num_masks))
