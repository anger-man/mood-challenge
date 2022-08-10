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

## Random convex hull
def contained(x):
    return np.all(np.asarray(x) @ A.T + b.T < 1e-5, axis =1)

N = 256
f = np.zeros([N, N, N])

x = np.linspace(-N/2, N/2-1, N)
X,Y,Z = np.meshgrid(x,x,x)

num_points = 0
max_points = 4

dist = 50

pointlist = []
point0x = random.randrange(int(N/2-1))
point0y = random.randrange(int(N/2-1))
point0z = random.randrange(int(N/2-1))
pointlist.append([point0x, point0y])
while True:
    pointx = random.randrange(int(N/2-1))
    pointy = random.randrange(int(N/2-1))
    pointz = random.randrange(int(N/2-1))
    if np.sqrt((point0x-pointx)**2 + (point0y-pointy)**2 + (point0z-pointz)**2) <= dist:
        pointlist.append([pointx, pointy])
        num_points += 1
    
    if num_points >= max_points:
        break

hull = ConvexHull(pointlist)
A, b = hull.equations[:,:-1], hull.equations[:, -1:]


x = []
y = []
for i in hull.vertices:
    x.append(pointlist[i][0])
    y.append(pointlist[i][1])

x.append(x[0])
y.append(y[0])

t = np.arange(len(x))
ti = np.linspace(0, t.max(), 10*t.size)

xi = interp1d(t, x, kind='cubic')(ti)
yi = interp1d(t, y, kind='cubic')(ti)

import matplotlib.path as mplPath
polypath = mplPath.Path(np.array([xi, yi]).T)

mask = np.zeros([N, N])
for m in range(N):
    for n in range(N):
        if polypath.contains_point([m, n]):
            mask[m,n] = 1

plt.figure()
plt.imshow(mask)
plt.show()
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
