# Python demo code for image super-resolution using sparse coding network

By Zhaowen Wang (zhawang@adobe.com), 2015

This code is for academic purpose only. Not for commercial/industrial activities.

# Required libraries
* Python 2.7
* OpenCV 2.4
* Matlab/pymatbridge (only for generating low-res images)

# Models
- 52: x2, 128 dictionary size, LISTA structure, no recurrent structure, SHLU activation
- 310: x2, cascade network for beyond x2 SR

# Note
Minor difference from the results reported in the ICCV paper may exist due to implementation details.

# Citation

    @inproceedings{wang2015deep,
      Author = {Wang, Zhaowen and Liu, Ding and Yang, Jianchao and Han, Wei and Huang, Thomas S},
      Title = {Deep Networks for Image Super-Resolution with Sparse Prior},
      Booktitle = {IEEE International Conference on Computer Vision (ICCV)},
      Year = {2015},
      Organization = {IEEE}
    }
Paper link: http://arxiv.org/abs/1507.08905
