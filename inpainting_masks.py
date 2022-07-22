import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors


N = 64
out = np.zeros([N, N])

a = 10
b = 10

x = np.linspace(-N/2, N/2-1, N)
y = np.linspace(-N/2, N/2-1, N)
X, Y = np.meshgrid(x, y)

idx = (((X**2)/(a**2) + (Y**2)/(b**2)) <= 1)

out[idx] = 1

plt.figure()
plt.imshow(out)
plt.show()


# %%

N = 64
out = np.zeros([N, N, N])

a=b=c= 10

x = np.linspace(-N/2, N/2-1, N)
y = np.linspace(-N/2, N/2-1, N)
z = np.linspace(-N/2, N/2-1, N)
X, Y, Z = np.meshgrid(x, y, z)

#idx = (((X**2)/(a**2) + (Y**2)/(b**2) + (Z**2)/(c**2)) <= 1)
idx = X**2 + Y**2 + Z**2 <= 30**2

print(np.where(idx)[0])

out[idx] = 1

fig, ax = plt.subplots(1, 3)
for i in range(3):
    im = ax[i].imshow(out[:,:,i])
    plt.colorbar(im, ax=ax[i])
plt.show()

# %%

prop = 0.25

while True:
    x_axis = a*np.random.rand(1)
    y_axis = b*np.random.rand(1)
    z_axis = c*np.random.rand(1)
    centre = np.array([N*np.random.rand(1), N*np.random.rand(1), N*np.random.rand(1)])

    if np.sqrt((centre[0]-N/2)**2 + (centre[1]-N/2)**2 + (centre[2]-N/2)**2) <= N/4:

        Xmove = X - centre[0]
        Ymove = Y - centre[1]
        Zmove = Z - centre[2]

        idx = (Xmove**2/x_axis**2 + Ymove**2/y_axis**2 + Zmove**2/z_axis**2) <= 1

        out[idx] = 1
        print('found it')         
        break

plt.figure()
plt.scatter(out[0], out[1], out[2])
plt.show()

# %%
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# number of ellipsoids 
ellipNumber = 2

# relative amout of volume occupied in unit cube
relVol = 0.25

#set colour map so each ellipsoid as a unique colour
norm = colors.Normalize(vmin=0, vmax=ellipNumber)
cmap = cm.jet
m = cm.ScalarMappable(norm=norm, cmap=cmap)

#compute each and plot each ellipsoid iteratively    

for indx in range(ellipNumber):
    # your ellispsoid and center in matrix form
    A = np.array([[np.random.random_sample(),0,0],
                  [0,np.random.random_sample(),0],
                  [0,0,np.random.random_sample()]])
    center = [indx*np.random.random_sample(),indx*np.random.random_sample(),indx*np.random.random_sample()]

    # find the rotation matrix and radii of the axes
    U, s, rotation = linalg.svd(A)
    radii = 1.0/np.sqrt(s) * 0.3 #reduce radii by factor 0.3 

    # calculate cartesian coordinates for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, 256)
    v = np.linspace(0.0, np.pi, 256)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center


    ax.plot_surface(x, y, z,  color=m.to_rgba(indx), linewidth=0.1, alpha=1, shade=True)
plt.show()




# %%

#def inpainting_masks(N, shape):

N = 256
a = 0.9
b = 1
c = 0.001

#def draw_ellipse(N, a, b, c):
rx = 1/np.sqrt(a)
ry = 1/np.sqrt(b)
rz = 1/np.sqrt(c)

u = np.linspace(0, 2*np.pi, N)
v = np.linspace(0, np.pi, N)

x = rx*np.outer(np.cos(u), np.sin(v))
y = ry*np.outer(np.sin(u), np.sin(v))
z = rz*np.outer(np.ones_like(u), np.cos(v))

    #return x, y, z


x, y, z = draw_ellipse(N, a, b, c)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z)
plt.show()

# %%
