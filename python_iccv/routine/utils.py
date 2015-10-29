#! /usr/bin/env python

import numpy as np
import cv2

def modcrop(im, modulo):
    sz = im.shape
    h = sz[0]/modulo*modulo
    w = sz[1]/modulo*modulo
    ims = im[0:h, 0:w, ...]
    return ims

def imresize(im_l, s):
    if s<1:
        im_l = cv2.GaussianBlur(im_l, (7,7), 0.5)
    im_h = cv2.resize(im_l, (0,0), fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return im_h

def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def ycbcr2rgb(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*255.0-16)/(235-16) #to [0, 1]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*255.0-16)/(240-16) #to [0, 1]
    im_ycrcb = im_ycbcr[:,:,(0,2,1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return im_rgb

def shave(im, border):
    if isinstance(border, int):
        border=[border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im

def ExtendBorder(im, offset):
    sz = im.shape
    assert(len(sz)==2)

    im2 = np.zeros([sz[0]+offset*2, sz[1]+offset*2])
    im2[ offset:-offset, offset:-offset ] = im
    im2[ offset:-offset, 0:offset ] = im[:, offset:0:-1]
    im2[ offset:-offset, -offset: ] = im[:, -2:-(offset+2):-1]
    im2[ 0:offset, :] = im2[2*offset:offset:-1, :]
    im2[ -offset:, :] = im2[-(offset+2):-(2*offset+2):-1, :]

    return im2

def ExtrConvFea(im, fltrs):
    """
    % extract convoluation features from whole image output
    % fea: [mxnxf], where f is the number of features used
    """
    m,n = im.shape
    nf = fltrs.shape[1] # 100
    fs = int(np.round(np.sqrt(fltrs.shape[0]))) # 9
    hfs = fs/2
    fea = np.zeros([m-fs+1, n-fs+1, nf]) # m - 9 + 1, n - 9 + 1 , 100
    for i in range(nf):
        fltr = fltrs[:, i].reshape([fs, fs])
        acts = cv2.filter2D(im, -1, fltr)
        fea[:, :, i] = acts[hfs:-hfs, hfs:-hfs]
    return fea

def ShLU(a, th):
    return np.sign(a)*np.maximum(0, np.abs(a)-th)

