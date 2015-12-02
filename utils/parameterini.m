function [network] = parameterini(folder_num)
%% Parameters
L = 5;
C = 5;
%% Dictionary
D_size = 128;
lr_size = 100;
hr_size = 25;
Dx = normrnd(0,1,[D_size,hr_size]);
Dy = normrnd(0,1,[D_size,lr_size]);

%% H layer
% H layer part I
H_layer_conv = ones(5,5,4);
% Haar-like filter
H_layer_conv(:,1:3,1) = -1;
H_layer_conv(1:3,:,2) = -1;
H_layer_conv(:,3,3) = -1;
H_layer_conv(1:3,4:5,4) = -1;
H_layer_conv(4:5,1:3,4) = -1;
% H layer part II
H_layer_pos  = zeros(5,5,25);
for i = 1:25
    temp = zeros(5,5);
    if mod(i,5)~=0
        temp(5 - ceil(i/5) + 1,5 - mod(i,5) + 1) = 1;
    else
        temp(5 - ceil(i/5) + 1,1) = 1;
    end
    H_layer_pos(:,:,i) = temp;
end
% combine I and II
H_layer = zeros(9,9,100);
for i = 1:100
    conv_pad = padarray(H_layer_conv(:,:,ceil(i/25)),[2,2]);
    if mod(i,25) ~= 0
        H_layer(:,:,i) = imfilter(conv_pad,H_layer_pos(:,:,mod(i,25)));
    else
        H_layer(:,:,i) = imfilter(conv_pad,H_layer_pos(:,:,25));
    end
end
network.conv1 = H_layer_conv;
network.conv2 = H_layer_pos;
network.conv = H_layer;
%% W,S,DX layer
network.w = C * Dy';
network.s = eye(D_size,D_size) - Dy * Dy';
network.dx = 1/C/L * Dx;

%% Diff
network.diffms = ones(5,5,25) / (-25);
for i = 1:25
    if mod(i,5) ~= 0
        network.diffms(mod(i,5),ceil(i/5),i) = 24/25;
    else
        network.diffms(5,ceil(i/5),i) = 24/25;
    end
end
network.diffms = padarray(network.diffms,[2,2]);
%% Mean
mean = load('mean.mat');
mean.mean2 = padarray(mean.mean2,[2,2]);
network.mean2 = mean.mean2;
%% G layer
network.addp = normrnd(0,1,[5,5]); 
%% other
network.theta = 1;
network.k = folder_num;
end

