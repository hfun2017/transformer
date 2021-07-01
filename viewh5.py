import h5py
import tkinter.filedialog
import numpy as np
from mayavi import mlab


def model2xyz(model):
    print(model.shape)
    return model[:,0],model[:,1],model[:,2]

fn=tkinter.filedialog.askopenfilename(title='选择文件', filetypes=[('h5','.h5')])
f=h5py.File(fn,"r")
cls_list=('ape','benchvise','cam','can','cat','driller','duck','eggbox','glue','holepuncher','iron','lamp','phone')

idx=np.random.randint(0,len(cls_list)-1)
model=f['duck'][idx]
x,y,z=model2xyz(model)
figure = mlab.figure(bgcolor=(1,1,1))
mlab.points3d(x,y,z,color=(0.3,0.3,1),figure=figure,scale_factor=0.02)

model=f['duck'][idx+1]
x,y,z=model2xyz(model)
figure = mlab.figure(bgcolor=(1,1,1))
mlab.points3d(x,y,z,color=(0.3,0.3,1),figure=figure,scale_factor=0.02)

mlab.show()