function im_h_y = SCN_Matconvnet(im_l_y, model,scale,use_gpu)

model_scale = 2;
patch_size = 5;
border_size = 6;
scale_y = 1.1;

% H layer: conv = conv_h1 * conv_h2
conv = single(reshape(model.conv,9,9,1,100));
conv_h1 = single(reshape(model.conv_h1,5,5,1,4));
conv_h2 = model.conv_h2;
mean2 = single(reshape(model.mean2,13,13,1,1));
diffm = single(reshape(model.diffms,9,9,1,25));
w = single(reshape(model.wd,1,1,100,128));
s = single(reshape(model.usd1,1,1,128,128));
dx = single(reshape(model.ud,1,1,128,25));
% G layer: conv_g1 * conv_g2
conv_g1 = single(reshape(model.addp,1,1,1,25));
conv_g2 = model.conv_g2;

iter_all = ceil(log(scale)/log(model_scale));
[lh,lw] = size(im_l_y);
for iter = 1:iter_all
    fprintf('iter:%d\n',iter);
    im_y = single(imresize(im_l_y,model_scale,'bicubic'));
    im_y = padarray(im_y,[border_size,border_size],'symmetric');
    
% we can replace im_y * conv by im_y * conv_h1 * conv_h2
% you can check it by comparing convfea_mat_2 and convfea_mat;
    convfea_mat_1 = vl_nnconv(im_y,conv_h1,[]);
    convfea_mat_2 = vl_nnconv(convfea_mat_1,conv_h2,[]);
    convfea_mat   = convfea_mat_2;
%     convfea_mat = vl_nnconv(im_y,conv,[]);

    im_mean_mat = vl_nnconv(im_y,mean2,[]);
    diffms_mat = vl_nnconv(im_y,diffm,[]);
    convfea_mat = vl_nnnormalize(convfea_mat,[size(convfea_mat,1),0,1,0.5]);
    wd_mat = vl_nnconv(convfea_mat,w,[]);
    z0_mat = ShlU(wd_mat,1);
    c1_mat = vl_nnconv(z0_mat,s,[]);
    c1_mat = c1_mat + wd_mat;
    z_mat = ShlU(c1_mat,1);  
    hPatch_mat = vl_nnconv(z_mat,dx,[]);
    hPatch_mat = vl_nnnormalize(hPatch_mat,[size(hPatch_mat,1),0,1,0.5]);
    mNorm_mat = vl_nnnormalize(diffms_mat,[size(diffms_mat,1),0,1,0.5]);
    mNorm_mat = mNorm_mat ./ diffms_mat;
    hPatch_mat = hPatch_mat ./ mNorm_mat * scale_y;
    hPatch_mat = vl_nnconv(hPatch_mat,conv_g1,[]);
    hPatch_mat = vl_nnconv(hPatch_mat,conv_g2,[]);
    im_h_y = im_mean_mat + hPatch_mat; 
    im_l_y = im_h_y;
end

if size(im_h_y,1) > lh * scale
   im_h_y = imresize(im_h_y,[lh * scale,lw * scale],'bicubic');
end
end
