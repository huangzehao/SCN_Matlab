load ./model/weights_srnet_x2_52
conv = reshape(model.conv,9,9,100);
diffms = reshape(model.diffms,9,9,25)
mean  = reshape(model.mean2,13,13,1) 