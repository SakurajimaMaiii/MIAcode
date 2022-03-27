# -*- coding: utf-8 -*-
import numpy as np
import torch
import math
import torch.nn.functional as F

def test_single_volume_slicebyslice(img,model):
    """
    test a single volume slice by slice using 2D network
    input: img 3d array must be [z,y,x] z is slice index
    output: array [z,y,x] with prediction
    """
    model.cuda()
    model.eval()
    
    num_slices = np.shape(img)[0]
    prediction_3d = []
    for i in range(num_slices):
        img2d = img[i]
        img2d = torch.from_numpy(img2d).unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad():
            pred_2d = model(img2d) 
            pred_2d = torch.argmax(pred_2d,1)
            pred_2d = pred_2d.detach().cpu().numpy()[0]
        prediction_3d.append(pred_2d)  
    prediction_3d = np.stack(prediction_3d)
    return prediction_3d

def test_single_case(net, image, stride, patch_size, num_classes=1):
    """
    predict 3d volume using slide window

    Parameters
    ----------
    net : model
    image : must be 3d array
    stride : 
    patch_size : 
    num_classes : number of class

    Returns
    -------
    label_map : prediction, shape is the same as image
    score_map : softmax outputs, shape [C,*]

    """
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride[0]) + 1
    sy = math.ceil((hh - patch_size[1]) / stride[1]) + 1
    sz = math.ceil((dd - patch_size[2]) / stride[2]) + 1
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride[0]*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride[1] * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride[2] * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map
