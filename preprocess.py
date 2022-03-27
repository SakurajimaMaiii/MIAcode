# -*- coding: utf-8 -*-
"""
useful functions for preprocess
"""

import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
import random


#load data
def read_nii(path):
    """
    path: data path
    if itkimg shape is [x,y,z], npimg shape is [z,y,x]
    """
    itkimg = sitk.ReadImage(path)
    npimg = sitk.GetArrayFromImage(itkimg)
    npimg = npimg.astype(np.float32)
    return npimg


def read_info(path):
    itkimg = sitk.ReadImage(path)
    spacing = itkimg.GetSpacing()
    origin = itkimg.GetOrigin()
    direction = itkimg.GetOrigin()
    return spacing,origin,direction

#normalization
def normalize_nonzero(img):
    mask = img.copy()
    mask[img>0] = 1
    mean = np.sum(mask*img) / np.sum(mask)
    std = np.sqrt(np.sum(mask * (img - mean)**2) / np.sum(mask))
    img = (img-mean)/std
    return img

def cut_window(img,low,high):
    #for CT data, cut HU value
    img = img.clip(img,low,high)
    return img

def normalize_to01(img):
    xmin = np.min(img)
    xmax = np.max(img)
    if xmax>xmin:
        img = (img-xmin)/(xmax-xmin)
    else:
        print('may const?')
        img = np.zeros(img.shape)
    return img


def copy_geometry(image: sitk.Image, ref: sitk.Image):
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image


#### resize 
def resize_segmentation(segmentation, new_shape, order=0, cval=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


def resample(img,old_spacing,new_spacing=(1.0,1.0,1.0),data_type='img'):
    #follow nn-unet
    # order=3 for data
    # order=0 for label
    
    shape = np.shape(img)
    new_shape = np.round(((np.array(old_spacing) / np.array(new_spacing)).astype(float) * shape)).astype(int)
    if data_type == 'img':
        new_data = resize(img,new_shape,order=3)
    elif data_type == 'seg':
        new_data = resize_segmentation(img,new_shape)
    else:
        raise NotImplementedError('data type is not implemented')
    return new_data


#crop

def center_crop_3d(data,crop_size=(128,128,128)):
    """
    center crop for 3d data
    input must be[C,D,H,W] or [D,H,W](will be expand to [1,D,H,W])
    output shape will be same as input.
    """
    dims = len(np.shape(data))
    if dims==3:
        data = np.expand_dims(data,axis=0)
    
    c,x,y,z = np.shape(data)
    #pad if necessary
    if x<=crop_size[0] or y<=crop_size[1] or z<=crop_size[2]:
        px = max((crop_size[0] - x) // 2 + 3, 0)
        py = max((crop_size[1] - y) // 2 + 3, 0)
        pz = max((crop_size[2] - z) // 2 + 3, 0)
        data = np.pad(data,[(0,0),(px,px),(py,py),(pz,pz)],mode='constant', constant_values=0)
        print('data size is smaller than crop size, we pad it')
        
    c,x,y,z = np.shape(data)
    x1 = int(round((x - crop_size[0]) / 2.))
    y1 = int(round((y - crop_size[1]) / 2.))
    z1 = int(round((z - crop_size[2]) / 2.))
    
    data = data[:,x1:x1+crop_size[0],y1:y1+crop_size[1],z1:z1+crop_size[2]]
    
    if dims==3:
        return data[0]
    return data


def center_crop_2d(data,crop_size=(128,128)):
    """
    center crop for 2d data
    input must be[C,H,W] or [H,W](will be expand to [1,H,W])
    output shape will be same as input.
    """
    dims = len(np.shape(data))
    if dims==2:
        data = np.expand_dims(data,axis=0)
    
    c,x,y = np.shape(data)
    #pad if necessary
    if x<=crop_size[0] or y<=crop_size[1]:
        px = max((crop_size[0] - x) // 2 + 3, 0)
        py = max((crop_size[1] - y) // 2 + 3, 0)
        data = np.pad(data,[(0,0),(px,px),(py,py)],mode='constant', constant_values=0)
        print('data size is smaller than crop size, we pad it')
        
    c,x,y = np.shape(data)
    x1 = int(round((x - crop_size[0]) / 2.))
    y1 = int(round((y - crop_size[1]) / 2.))
    
    data = data[:,x1:x1+crop_size[0],y1:y1+crop_size[1]]
    
    if dims==2:
        return data[0]
    return data

def random_crop_3d(data,crop_size=(128,128,128)):
    """
    random crop for 3d data
    input must be[C,D,H,W] or [D,H,W](will be expand to [1,D,H,W])
    output shape will be same as input.
    """
    dims = len(np.shape(data))
    if dims==3:
        data = np.expand_dims(data,axis=0)
    
    c,x,y,z = np.shape(data)
    #pad if necessary
    if x<=crop_size[0] or y<=crop_size[1] or z<=crop_size[2]:
        px = max((crop_size[0] - x) // 2 + 3, 0)
        py = max((crop_size[1] - y) // 2 + 3, 0)
        pz = max((crop_size[2] - z) // 2 + 3, 0)
        data = np.pad(data,[(0,0),(px,px),(py,py),(pz,pz)],mode='constant', constant_values=0)
        print('data size is smaller than crop size, we pad it')
    
    c,x,y,z = np.shape(data)
    x1 = np.random.randint(0,x-crop_size[0])
    y1 = np.random.randint(0,y-crop_size[1])
    z1 = np.random.randint(0,z-crop_size[2])
    
    data = data[:,x1:x1+crop_size[0],y1:y1+crop_size[1],z1:z1+crop_size[2]]
    
    if dims==3:
        return data[0]
    return data


def random_crop_2d(data,crop_size=(128,128)):
    """
    random crop for 3d data
    input must be[C,H,W] or [H,W](will be expand to [1,H,W])
    output shape will be same as input.
    """
    dims = len(np.shape(data))
    if dims==2:
        data = np.expand_dims(data,axis=0)
    
    c,x,y = np.shape(data)
    #pad if necessary
    if x<=crop_size[0] or y<=crop_size[1]:
        px = max((crop_size[0] - x) // 2 + 3, 0)
        py = max((crop_size[1] - y) // 2 + 3, 0)
        data = np.pad(data,[(0,0),(px,px),(py,py)],mode='constant', constant_values=0)
        print('data size is smaller than crop size, we pad it')
    
    c,x,y = np.shape(data)
    x1 = np.random.randint(0,x-crop_size[0])
    y1 = np.random.randint(0,y-crop_size[1])
    
    data = data[:,x1:x1+crop_size[0],y1:y1+crop_size[1]]
    
    if dims==2:
        return data[0]
    return data



def random_flip(data,p=0.5):
    """
    random flip for 3d data
    input shape must be [C,D,H,W]
    """
    if random.random() < p:
        data = np.flip(data,axis=1)
    if random.random() < p:
        data = np.flip(data,axis=2)
    if random.random() < p:
        data = np.flip(data,axis=3)
    return data

    
    
    







