#! /usr/bin/env python

import Image
import numpy as np
import cPickle as pickle
import utils

class SRBase(object):
    def __init__(self):
        pass

    def upscale(self, im_l, s):
        """
        % im_l: LR image, float np array in [0, 255]
        % im_h: HR image, float np array in [0, 255]
        """
        im_l = im_l/255.0
        if len(im_l.shape)==3 and im_l.shape[2]==3:
            im_l_ycbcr = utils.rgb2ycbcr(im_l)
        else:
            im_l_ycbcr = np.zeros([im_l.shape[0], im_l.shape[1], 3])
            im_l_ycbcr[:, :, 0] = im_l
            im_l_ycbcr[:, :, 1] = im_l
            im_l_ycbcr[:, :, 2] = im_l

        im_l_y = im_l_ycbcr[:, :, 0]*255 #[16 235]
        im_h_y = self.upscale_alg(im_l_y, s)

        # recover color
        if len(im_l.shape)==3:
            im_ycbcr = utils.imresize(im_l_ycbcr, s);
            im_ycbcr[:, :, 0] = im_h_y/255.0; #[16/255 235/255]
            im_h = utils.ycbcr2rgb(im_ycbcr)*255.0
        else:
            im_h = im_h_y

        im_h = np.clip(im_h, 0, 255)
        im_h_y = np.clip(im_h_y, 0, 255)
        return im_h,im_h_y

    def upscale_alg(self, im_l_y, s):
        pass

class Bicubic(SRBase):
    def upscale_alg(self, im_l_y, s):
        im_h_y = utils.imresize(im_l_y, s)
        return im_h_y

class SCN(SRBase):
    def __init__(self, model_files):
        self.mdls = []
        for f in model_files:
            self.mdls += [pickle.load(open(f, 'rb'))]
        i=model_files[0].find('_x')
        self.MDL_SCALE = int(model_files[0][i+2]);
        self.PATCH_SIZE = 5
        self.BORDER_SIZE = 6
        self.SCALE_Y = 1.1 #linear factor on scaley layer

    def upscale_alg(self, im_l_y, s):
        h_gt, w_gt = im_l_y.shape[0]*s, im_l_y.shape[1]*s
        hpsz = self.PATCH_SIZE/2

        itr_all = int(np.ceil(np.log(s)/np.log(self.MDL_SCALE)))
        for itr in range(itr_all):
            print 'itr:', itr
            im_y = utils.imresize(im_l_y, self.MDL_SCALE)
            im_y = utils.ExtendBorder(im_y, self.BORDER_SIZE)
            mdl=self.mdls[itr]

            # extract gradient features
            convfea = utils.ExtrConvFea(im_y, mdl['conv'])
            im_mean = utils.ExtrConvFea(im_y, mdl['mean2'])
            diffms = utils.ExtrConvFea(im_y, mdl['diffms'])

            # matrix operation
            h, w, c = convfea.shape
            convfea = convfea.reshape([h*w, c])
            convfea_norm = np.linalg.norm(convfea, axis=1)
            convfea = (convfea.T/convfea_norm).T
            wd = np.dot(convfea, mdl['wd'])
            z0 = utils.ShLU(wd, 1)
            z = utils.ShLU(np.dot(z0, mdl['usd1'])+wd, 1) #sparse code

            hPatch = np.dot(z, mdl['ud'])
            hNorm = np.linalg.norm(hPatch, axis=1)
            diffms = diffms.reshape([h*w, diffms.shape[2]])
            mNorm = np.linalg.norm(diffms, axis=1)
            hPatch = (hPatch.T/hNorm*mNorm).T*self.SCALE_Y
            hPatch = hPatch*mdl['addp'].flatten()

            hPatch = hPatch.reshape([h, w, hPatch.shape[1]])
            im_h_y = im_mean[:, :, 0]
            h, w = im_h_y.shape
            cnt = 0
            for ii in range(self.PATCH_SIZE-1, -1, -1):
                for jj in range(self.PATCH_SIZE-1, -1, -1):
                    im_h_y = im_h_y+hPatch[jj:(jj+h), ii:(ii+w), cnt]
                    cnt = cnt+1
            
            im_l_y = im_h_y

        # shrink size to gt
        if (im_h_y.shape[0]>h_gt):
            print 'downscale from {} to {}'.format(im_h_y.shape, (h_gt, w_gt))
            im_h_y = utils.imresize(im_h_y, 1.0*h_gt/im_h_y.shape[0])
            assert(im_h_y.shape[1]==w_gt)

        return im_h_y


