import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from scipy import ndimage, misc

N = 256
f = np.zeros([N, N, N])

prop = 0.0001
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
