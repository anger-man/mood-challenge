import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from scipy import ndimage, misc
import random
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import alphashape

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


def create_masks(N, max_points, dist, it):
    x = np.linspace(0, 1, N)
    X,Y,Z = np.meshgrid(x,x,x)

    pointslist = get_points(N, dist, max_points)
    points = []
    [points.append(x) for x in pointslist if x not in points]

    #x = []
    #y = []
    #z = []
    #for i in range(len(points)):
    #    x.append(points[i][0])
    #    y.append(points[i][1])
    #    z.append(points[i][2])

    #x.append(x[0])
    #y.append(y[0])
    #z.append(z[0])

    #t = np.arange(len(x))
    #ti = np.linspace(0, t.max(), 10*t.size)

    #xi = interp1d(t, x, kind='quadratic')(ti)
    #yi = interp1d(t, y, kind='quadratic')(ti)
    #zi = interp1d(t, z, kind='quadratic')(ti)

    #pointss = []
    #for i in range(len(xi)):
    #    pointss.append([xi[i], yi[i], zi[i]])

    alpha_shape = alphashape.alphashape(points, 2.0)

    x = np.linspace(0, 1, N)
    X, Y, Z = np.meshgrid(x,x,x)
    mask = np.zeros([N, N, N])
    total = N*N*N
    steps = 1
    for m in range(N):
        xm = x[m]
        for n in range(N):
            xn = x[n]
            for l in range(N):
                #tmp_point = Point(x[m], x[n], x[l])
                tmp_point = [(xm,xn,x[l])]
                if alpha_shape.contains(tmp_point):
                #if contained([m, n, l]):
                    mask[m,n,l] = 1
                #print('Completed ' + str(round(steps/total*100)) + '%')
                steps += 1

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
    np.save('masks/arrays/' + name, mask.astype(np.uint8))
    plt.savefig('masks/plots/' + name)
    plt.close()

    return mask


N = 256
max_points = 1000
num_masks = 1

max_size = 0.15
min_size = 0.05

for it in range(num_masks):
    distrand = np.random.rand()
    dist = distrand*(max_size-min_size) + min_size
    mask = create_masks(N, max_points, dist, it)
    print(str(it) + '/' + str(num_masks))

# %%

pointss = [tuple(x) for x in points]

mask = np.zeros([N, N, N])
total = N*N*N
steps = 1
for m in range(N):
    for n in range(N):
        for l in range(N):
            tmp_point = [(m, n, l)]
            if alpha_shape.contains(tmp_point):
            #if contained([m, n, l]):
                mask[m,n,l] = 1
            print('Completed ' + str(round(steps/total*100)) + '%')
            steps += 1

plt.show()
# %%

import matplotlib.pyplot as plt
from descartes import PolygonPatch

import alphashape
import random
from shapely.geometry import Point
import pyglet

N = 128
dist = 0.10
max_points = 30

#points = [(0., 0., 0.9), (0., 1., 0.1), (1., 1., 1.), (1., 0., 0.25),(0.5, 0.25, 0.5), (0.5, 0.75, 0.75), (0.25, 0.5, 0.1), (0.75, 0.5, 0.3)]
#alpha_shape = alphashape.alphashape(points, lambda ind, r: 1.0 + any(np.array(points)[ind][:,0] == 0.0)) # Create the alpha shape
alpha_shape = alphashape.alphashape(pointss, 2.0)

# %%
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
plt.show()

# %%
# Plotting the alpha shape over the input data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

N = 10 # number of random points
for i in range(N):
    x = round(random.uniform(0, 1), 2)
    y = round(random.uniform(0, 1), 2)
    z = round(random.uniform(0, 1), 2)

    point = [(x,y,z)] # analysis point

    if alpha_shape.contains(point) == True:
        plt.scatter(x,y,z,c='blue')
    else:
        plt.scatter(x,y,z,c='red')

plt.show()
