train_data_path = './data/train/BSDS300/images/train';
type = '*.jpg';
up_scale = 2;
im_cropsize = [56,56];

im_dir = dir(fullfile(train_data_path,type));
im_num = length(im_dir);
train_lr_data = zeros(im_cropsize(1),im_cropsize(2),im_num);
train_hr_data = zeros(im_cropsize(1),im_cropsize(2),im_num);

for ii = 1:im_num
    im = imread(fullfile(train_data_path,im_dir(ii).name));
    im = double(im)/255.0;
    im_ycbcr = rgb2ycbcr(im);
    im = im_ycbcr(:,:,1);
    [h,w] = size(im);
    im = im(floor(h/2-im_cropsize(1)/2):floor(h/2-im_cropsize(1)/2)+im_cropsize(1)-1,floor(w/2-im_cropsize(2)/2):floor(w/2-im_cropsize(2)/2)+im_cropsize(2)-1);
    train_hr_data(:,:,ii) = im;
    im = imresize(im,1/up_scale,'bicubic');
    im = imresize(im,im_cropsize,'bicubic');
    train_lr_data(:,:,ii) = im;
end
save('train_lr_data.mat','train_lr_data');
save('train_hr_data.mat','train_hr_data');