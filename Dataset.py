
# coding: utf-8

# In[7]:

import os
import matplotlib.gridspec as gridspec
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import cv2

class DataProvider():
    def __init__(self):
        self.cars=None
        

    def deserialize(self,path, fext='.png'):
        fnames = []
        #print(parent)
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(fext):
                    fnames.append(os.path.join(root, f))
        return fnames
    
    def gridOfrandomimages(self,fnames, rows=6,cols=6,title=None):
        nbimages = rows*cols
        #print(nbimages)
        rndfnames = np.random.choice(fnames,nbimages )
        images = []
        for fname in rndfnames:
            images.append(img.imread(fname))

        grid_space = gridspec.GridSpec(cols, rows)
        grid_space.update(wspace=0.1, hspace=0.1)
        plt.figure(figsize=(rows, cols))

        for idx in range(0, nbimages):
            axis_1 = plt.subplot(grid_space[idx])
            axis_1.axis('off')
            axis_1.imshow(images[idx])

        if title is not None:
            plt.suptitle(title)
        plt.show()
        
        
    @staticmethod
    def files(path,fext='.png', show=True):
        dt = DataProvider()
        fnames = dt.deserialize(path,fext)
        if show :
            dt.gridOfrandomimages(fnames)
        return fnames

    @staticmethod
    def visualize(fnames):
        dt = DataProvider()
        dt.gridOfrandomimages(fnames)
    


# In[8]:

DataProvider.files('./vehicles')


# In[ ]:



