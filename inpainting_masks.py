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

# %% 
## Random convex hull
def contained(x):
    return np.all(np.asarray(x) @ A.T + b.T < 1e-5, axis =1)

def check_coplanar(points):
   
    p0 = np.array(points[0])
    p1 = np.array(points[1])
    p2 = np.array(points[2])

    D = np.linalg.det([p1-p0, p2-p0])
    return np.any(D)

def get_points(N, dist, max_points):
    num_points = 1
    pointlist = []
    lr = int(N/4)
    tr = int(3*N/4)
    point0x = random.randrange(lr, tr)
    point0y = random.randrange(lr, tr)
    point0z = random.randrange(lr, tr)
    pointlist.append([point0x, point0y, point0z])
    while True:
        pointx = random.randrange(lr, tr)
        pointy = random.randrange(lr, tr)
        pointz = random.randrange(lr, tr)
        if np.sqrt((point0x-pointx)**2 + (point0y-pointy)**2 + (point0z-pointz)**2) <= dist:
            pointlist.append([pointx, pointy, pointz])
            print('Points found ' + str(num_points) + '/' + str(max_points))
            num_points += 1
        
        if num_points > max_points:
            break

    return pointlist

N = 256
f = np.zeros([N, N, N])

x = np.linspace(-N/2, N/2-1, N)
X,Y,Z = np.meshgrid(x,x,x)

max_points = 500
dist = 50

points = get_points(N, dist, max_points)
hull = ConvexHull(points)
A, b = hull.equations[:,:-1], hull.equations[:, -1:]
#print(check_coplanar(points))

x = []
y = []
z = []
for i in range(len(points)):
    x.append(points[i][0])
    y.append(points[i][1])
    z.append(points[i][2])

x.append(x[0])
y.append(y[0])
z.append(z[0])

t = np.arange(len(x))
ti = np.linspace(0, t.max(), 10*t.size)

xi = interp1d(t, x, kind='cubic')(ti)
yi = interp1d(t, y, kind='cubic')(ti)
zi = interp1d(t, z, kind='cubic')(ti)

pointss = []
for i in range(len(xi)):
    pointss.append([xi[i], yi[i], zi[i]])

import alphashape

alpha_shape = alphashape.alphashape(points[1:4], 2.0)
mask = np.zeros([N, N, N])
total = N*N*N
steps = 1
for m in range(N):
    for n in range(N):
        for l in range(N):
            if alpha_shape.contains(np.expand_dims(np.asarray([m, n, l]), axis=0)):
            #if contained([m, n, l]):
                mask[m,n,l] = 1
            print('Completed ' + str(round(steps/total*100)) + '%')
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
plt.show()
# %%

import matplotlib.path as mplPath
polypath = mplPath.Path(np.array([xi, yi]).T)

mask = np.zeros([N, N])
for m in range(N):
    for n in range(N):
        if polypath.contains_point([m, n]):
            mask[m,n] = 1

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xi, yi, zi)
plt.show()
# %%
plt.figure()
plt.imshow(mask)
plt.show()
# %%

##
fig, ax = plt.subplots()
ax.plot(xi, yi)
ax.plot(x, y)
plt.show()

##



## Random ellipses
N = 256
f = np.zeros([N, N, N])

prop = 0.001
x = np.linspace(-N/2, N/2-1, N)
X,Y,Z = np.meshgrid(x,x,x)

a = 25
b = 10
c = 40

while True:
    aa = a*np.random.rand(1)
    bb = b*np.random.rand(1)
    cc = c*np.random.rand(1)

    X_shift = X - np.sign(0.5 - np.random.rand(1))*N*np.random.rand(1)
    Y_shift = Y - np.sign(0.5 - np.random.rand(1))*N*np.random.rand(1)
    Z_shift = Z - np.sign(0.5 - np.random.rand(1))*N*np.random.rand(1)

    idx = (X_shift**2/aa**2 + Y_shift**2/bb**2 + Z_shift**2/bb**2) <= 1
    angle = np.random.randint(1, 90+1)
    idx = ndimage.rotate(idx, angle, reshape = False)
   
    f[idx] = 1

    fflat = f.flatten()

    if np.count_nonzero(fflat)/np.prod(f.shape) >= prop:
        break


s1 = np.squeeze(np.sum(f, 0))
s2 = np.squeeze(np.sum(f, 1))
s3 = np.squeeze(np.sum(f, 2))

s1[s1!=0] = 1
s2[s2!=0] = 1
s3[s3!=0] = 1

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(s1)
ax2.imshow(s2)
ax3.imshow(s3)
plt.show()
