close all
clear
addpath ./code
addpath ./data
addpath ./admm

%source: moving image
%target: fixed image

levels     = [4,2,1];
maxIter    = 2000;
tolerance  = 1e-2;
difference = 2;
talyor     = 5;
lambda     = .2;
mode       = 'tv';
padNum     = 8;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img2 = double(imread('lenag2.png'));
img1 = double(imread('lenag1.png'));
target = rescale_intensity(img2(:,:,1), [1, 99]);
source = rescale_intensity(img1(:,:,1), [1, 99]);
figure; imagesc(source); colormap(gray); title('original source image'); axis off; axis equal;
figure; imagesc(target); colormap(gray); title('original target image'); axis off; axis equal;

target = padarray(target,  [padNum,padNum], 0);
source = padarray(source,  [padNum,padNum], 0);
[u0, v0] = pyramid_flow(source, target, levels, talyor, maxIter, lambda, tolerance, difference, mode);
show_color_displacement(u0, v0, padNum, 1)
show_warped_image(source, u0, v0, padNum, 1, 'linear');
show_deformed_grid_direct(source, u0, v0, padNum, 3, 'g', 1)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
levels     = [4,2,1];
maxIter    = 2000;
tolerance  = 1e-3;
difference = 2;
talyor     = 5;
lambda     = .15;
mode       = 'sotv';
padNum     = 8;
S          = 5;

img = double(niftiread('sa_crop.nii.gz'));
target = rescale_intensity(img(:,:,S,1), [1, 99]);
source = rescale_intensity(img(:,:,S,16), [1, 99]);
figure; imagesc(source); colormap(gray); title('original source image'); axis off; axis equal;
figure; imagesc(target); colormap(gray); title('original target image'); axis off; axis equal;

target = padarray(target,  [padNum,padNum], 0);
source = padarray(source,  [padNum,padNum], 0);
[u0, v0] = pyramid_flow(source, target, levels, talyor, maxIter, lambda, tolerance, difference, mode);
show_color_displacement(u0, v0, padNum, 1)
show_warped_image(source, u0, v0, padNum, 1, 'linear');
show_deformed_grid_direct(source, u0, v0, padNum, 3, 'g', 1)

