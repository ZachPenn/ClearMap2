import holoviews as hv
import numpy as np
import SimpleITK as sitk
import skimage
from skimage import io
import os


def load_alignment_images(ws):
    
    sig = skimage.io.imread(ws.filename('resampled'))

    ref = sitk.ReadImage(
            os.path.join(ws.filename('auto_to_reference'),'result.1.mhd'))
    ref = sitk.GetArrayFromImage(ref)

    auto = sitk.ReadImage(
        os.path.join(ws.filename('resampled_to_auto'),'result.0.mhd'))
    auto = sitk.GetArrayFromImage(auto)
    return sig, auto, ref


def plane_display(p,img,plane,lims,title):
    
    if plane == 'h':
        array = img[p,:,:]
    elif plane == 'c':
        array = img[:,p,:]
    elif plane == 's':
        array = img[:,:,p]
    
    image = hv.Image((np.arange(array.shape[1]),np.arange(array.shape[0]),array))
    image.opts(
        width = int(array.shape[1]),
        height = int(array.shape[0]),
        invert_yaxis = True,
        clim = lims,
        title = title,
        cmap='gray',
        tools=['hover'])
    return image


def gen_hmap(img,plane,title=None,inter=25):
    
    if plane == 'h':
        idx = 0
    elif plane == 'c':
        idx = 1
    elif plane == 's':
        idx = 2
    
    lims = (0,int(img.max()))
        
    img_dict = {p : plane_display(p,img,plane,lims,title) for p in np.arange(0,img.shape[idx],inter)}
    return hv.HoloMap(img_dict, kdims='slice')