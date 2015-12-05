test_data_path = './data/train/BSDS300/images/test';
type = '*.jpg';
up_scale = 2;
im_cropsize = [56,56];

im_dir = dir(fullfile(test_data_path,type));
im_num = length(im_dir);
test_lr_data = zeros(im_cropsize(1),im_cropsize(2),im_num);
test_hr_data = zeros(im_cropsize(1),im_cropsize(2),im_num);

for ii = 1:im_num
    im = imread(fullfile(test_data_path,im_dir(ii).name));
    im = double(im)/255.0;
    im_ycbcr = rgb2ycbcr(im);
    im = im_ycbcr(:,:,1);
    [h,w] = size(im);
    im = im(floor(h/2-im_cropsize(1)/2):floor(h/2-im_cropsize(1)/2)+im_cropsize(1)-1,floor(w/2-im_cropsize(2)/2):floor(w/2-im_cropsize(2)/2)+im_cropsize(2)-1);
    test_hr_data(:,:,ii) = im;
    im = imresize(im,1/up_scale,'bicubic');
    im = imresize(im,im_cropsize,'bicubic');
    test_lr_data(:,:,ii) = im;
end
save('test_lr_data.mat','test_lr_data');
save('test_hr_data.mat','test_hr_data');