#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# # First, an intro to some handy commands in jupyter:
# ### (Demo to follow in class)
# From the first lecture, you may recall that instead of clicking the "run" button with your mouse, you can run a cell with "shift-enter". Jupyter is full of keyboard shortcuts like this. First, we need to know there are two modes to jupyter: command mode and edit mode.

# ## Edit mode:
# Edit mode is the one you're likely already familiar with. When you type code, you're in edit mode and all of the usual commands (cmd/ctrl + c to copy, cmd/ctl + v to paste, etc.) work as they would in a word document. A couple of extra commands that are very handy:
#
# - Typing ctrl (or fn on certain computers) + left/right arrow will take you the start of a line of code. Why is this useful? Imagine you typed out a line of code and your cursor is at the end of the code line.
#         my_data[10:30]+my_other_data[10:30]
#     You then realize you want to plot it, so you can either click on the far left of the code block to type plt.plot( or you can hit ctrl+left to jump all the way to the left of the line of code
# - Comment/uncomment code lines. Select multiple lines of code and hit cmd (ctrl on windows) + "/" (or "?") to comment and uncomment.
# - Multiselect. You can left click while holding cmd (ctrl on windows) to select multiple cursor locations in your code. If you had:
#         function1(my_data)
#         function2(my_data)
#         my_new_data = my_data + 7
#     You could replace my_data all at the same time by selecting multiple lines.
#
#
#
# When in Edit mode, press escape to enter command mode
# ## Command mode:
# - Press Enter to re-enter edit mode
# - "A" makes a new cell above the current cell selected
# - "B" makes a new cell below the current cell selected
# - pressing "D" twice deletes a cell
# - "F" enters find-and-replace mode for that cell only
# - "C" copies a cell
# - "V" pastes a cell below
# - many more! Type "H" to see them all

# In[2]:


body_ = pd.read_csv('3d_body.csv',header=None).to_numpy()
### We use the .to_numpy() to open the csv as a np array instead of a pd DataFrame.

### Generally speaking, pandas dataframes are useful when you want to run general statistics & explore
### a dataset, whereas numpy arrays are more useful objects to do math (like PCA) on


# In[3]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d',xlabel="x", ylabel="y", zlabel="z",
                     xlim=(-0.4, 0.8), ylim=(-0.4, 0.8), zlim=(-0.4, 0.8))
                        ### Here, we're using the object-oriented form of plotting in python
                        ### The control is superior and you can specify everything ahead of time
### An aside:
### Here, we want to plot the scatter of the data
### One way to do this is by looping. This is commented out because it's very slow!
# for i in body_[:20]:
#    ax.scatter(i[0],i[1],i[2],s=5,color="C0")

### A far more efficient way is to plot all points at once:
ax.scatter(*body_.T,s=5)
### .T transposes from ((x1,y1,z1),(x2,y2,z2),...) into ((x1,x2...),(y1,y2...),(z1,z2...))
### The * is syntactical sugar unpacks the x,y,z arrays as three inputs to this function

ax.view_init(20, -30)
plt.tight_layout()
fig.show()


# In[4]:


### Next, we'll demean this array so it's centered on 0
body = body_-np.mean(body_,axis=0)


# In[5]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d',xlabel="x", ylabel="y", zlabel="z",
                     xlim=(-0.4,0.4),ylim=(-0.4,0.4),zlim=(-0.4,0.4))
ax.scatter(*body.T,s=5)
ax.view_init(20, -30)
plt.tight_layout()


# In[6]:


### We'll also be working with rotated data. Rotating and plotting these data:
theta_x,theta_y,theta_z = np.deg2rad([30,60,45]) ### Degrees of rotation along each axis

### Create rotation matrices:
Rx = np.array([[1,0,0],[0,np.cos(theta_x),np.sin(theta_x)],[0,-np.sin(theta_x),np.cos(theta_x)]])
Ry = np.array([[np.cos(theta_y),0,-np.sin(theta_y)],[0,1,0],[np.sin(theta_y),0,np.cos(theta_y)]])
Rz = np.array([[np.cos(theta_z),np.sin(theta_z),0],[-np.sin(theta_z),np.cos(theta_z),0],[0,0,1]])
### Apply the rotations (order matters!)
bodyR = ((body@Rx.T)@Ry.T)@Rz.T

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d',xlabel="x", ylabel="y", zlabel="z",
                     xlim=(-0.4,0.4),ylim=(-0.4,0.4),zlim=(-0.4,0.4))
ax.scatter(*bodyR.T,s=5)
ax.view_init(20, -30)
plt.tight_layout()
fig.show()


# In[7]:


### Variance calculation on the data prior to PCA. First for the unrotated data,
tot_var1 = np.var(body,axis=0)
total_var1 = sum(tot_var1)
frac1 = tot_var1/sum(tot_var1)
print("Unrotated explained variance fractions:",frac1)

### Then for the rotated data
tot_var2 = np.var(bodyR,axis=0)
total_var2 = sum(tot_var2)
frac2 = tot_var2/sum(tot_var2)
print("Rotated explained variance fractions:", frac2)


# In[8]:


### We'll first do PCA on the unrotated data
pca = PCA(n_components = 3)
PCs = pca.fit_transform(body) # find PCs
eigvecs = pca.components_  # find eignevectors
fracVar = pca.explained_variance_ratio_  # give fraction of variance per model
B = pca.transform(body)


# In[9]:


### Plotting the eigenvectors and PCs of the first three modes:
fig = plt.figure(figsize=(8,6))
for i in range(3):
    ax = fig.add_subplot(3,2,2*i+1,xticks=[0,1,2],xticklabels=["x","y","z"],ylabel=f"e{i+1}",
                         title=f"variance explained = {round(fracVar[i]*100,2)}%")
    ax.plot(eigvecs[i],marker="o")
    ax = fig.add_subplot(3,2,2*i+2,ylabel=f"PC{i+1}",ylim=(-0.4,0.4))
    ax.scatter(np.arange(len(PCs.T[i])),PCs.T[i],s=1)

plt.tight_layout()


# In[10]:


### Lets plot the data in the space of these eigenvectors:
### PC1 vs PC2, PC1 vs PC3, and PC2 vs PC3

fig = plt.figure(figsize=(15,4))


ax1 = fig.add_subplot(131,xlabel="PC2",ylabel="PC1",xlim=(-0.4,0.4),ylim=(-0.4,0.4))
ax1.scatter(PCs.T[1],PCs.T[0],s=1)
ax2 = fig.add_subplot(132,xlabel="PC3",ylabel="PC1",xlim=(-0.4,0.4),ylim=(-0.4,0.4))
ax2.scatter(PCs.T[2],PCs.T[0],s=1)
ax3 = fig.add_subplot(133,xlabel="PC3",ylabel="PC2",xlim=(-0.4,0.4),ylim=(-0.4,0.4))
ax3.scatter(PCs.T[2],PCs.T[1],s=1)


# In[11]:


### lets plot the data in the original 3-D space (x,y,z) and then add the eigenvectors plotted as vectors
### for vector plotting we use plt.quiver function
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d',xlabel="x", ylabel="y", zlabel="z",
                     xlim=(-0.4,0.4),ylim=(-0.4,0.4),zlim=(-0.4,0.4))
ax.scatter(*body.T,s=5,alpha=0.2)
ax.view_init(20, -30)
for i in range(3):
    plt.quiver(*[0]*3,*eigvecs[i]*0.2,color=f"C{i+1}",zorder=-100,arrow_length_ratio=0.2,lw=2)

plt.tight_layout()
plt.show()

# In[12]:


### Now, doing PCA on the rotated data:
pcaR = PCA(n_components = 3)
PCsR = pcaR.fit_transform(bodyR)
eigvecsR = pcaR.components_
fracVarR = pcaR.explained_variance_ratio_


# In[14]:


### Looking at the first three eigenvectors and PCs:
fig = plt.figure(figsize=(8,6))
for i in range(3):
    ax = fig.add_subplot(3,2,2*i+1,xticks=[0,1,2],xticklabels=["x","y","z"],ylabel="e"+str(i+1),
                         title="variance explained = "+str(round(fracVarR[i]*100,2))+"%")
    ax.plot(eigvecsR[i],marker="o")
    ax = fig.add_subplot(3,2,2*i+2,ylabel="PC"+str(i+1),ylim=(-0.4,0.4))
    ax.scatter(np.arange(len(PCsR.T[i])),PCsR.T[i],marker="o",s=1)

plt.tight_layout()
fig.show()


# In[15]:


# same plots as before
fig = plt.figure(figsize=(15,7))
ax1 = fig.add_subplot(231,xlabel="y$_R$",ylabel="x$_R$",xlim=(-0.4,0.4),ylim=(-0.4,0.4))
ax1.scatter(bodyR.T[1],bodyR.T[0],s=1)
ax2 = fig.add_subplot(232,xlabel="z$_R$",ylabel="x$_R$",xlim=(-0.4,0.4),ylim=(-0.4,0.4))
ax2.scatter(bodyR.T[2],bodyR.T[0],s=1)
ax3 = fig.add_subplot(233,xlabel="z$_R$",ylabel="y$_R$",xlim=(-0.4,0.4),ylim=(-0.4,0.4))
ax3.scatter(bodyR.T[2],bodyR.T[1],s=1)


ax4 = fig.add_subplot(234,xlabel="PC2",ylabel="PC1",xlim=(-0.4,0.4),ylim=(-0.4,0.4))
ax4.scatter(PCs.T[1],PCs.T[0],s=1)
ax5 = fig.add_subplot(235,xlabel="PC3",ylabel="PC1",xlim=(-0.4,0.4),ylim=(-0.4,0.4))
ax5.scatter(PCs.T[2],PCs.T[0],s=1)
ax6 = fig.add_subplot(236,xlabel="PC3",ylabel="PC2",xlim=(-0.4,0.4),ylim=(-0.4,0.4))
ax6.scatter(PCs.T[2],PCs.T[1],s=1)
plt.tight_layout()
fig.show()

