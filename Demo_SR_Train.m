%  clear all;
%  close all;
%% Prepare
run matconvnet/matlab/vl_setupnn;
addpath('utils');
addpath('model');

% load train and test data
test_hr_data = load('./data/test_hr_data.mat');test_hr_data = test_hr_data.test_hr_data;
test_lr_data = load('./data/test_lr_data.mat');test_lr_data = test_lr_data.test_lr_data;
train_hr_data = load('./data/train_hr_data.mat');train_hr_data = train_hr_data.train_hr_data;
train_lr_data = load('./data/train_lr_data.mat');train_lr_data = train_lr_data.train_lr_data;

% network ini
folder_num = 1;
network = SCN_ini(folder_num);