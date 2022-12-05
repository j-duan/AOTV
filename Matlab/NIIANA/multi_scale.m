function multi_scale()

close all

% I1=imread('yos9.tif');
% I2=imread('yos10.tif');
% I1=I1(1:240,1:304);
% I2=I2(1:240,1:304);

I2=imread('1.png');
I1=imread('2.png');
I1 = I1(60:243,219:450);
I2 = I2(60:243,219:450);

% Convert images to grayscale
if size(size(I1),2)==3
    I1=rgb2gray(I1);
end
if size(size(I2),2)==3
    I2=rgb2gray(I2);
end
I1=double(I1);
I2=double(I2);

I1 = (I1-min(I1(:))) ./ (max(I1(:)) - min(I1(:)));
I2 = (I2-min(I2(:))) ./ (max(I2(:)) - min(I2(:)));

im1 = I1;
im2 = I2;

figure; imshow(I1,[]); title('frame 1');
figure; imshow(I2,[]); title('frame 2');
figure; imshowpair(I1,I2); title('before optical flow');

I1 = GaussianReplicateBoundary(I1,0.5);
I2 = GaussianReplicateBoundary(I2,0.5);

levels = [8,4,2,1]; 

u=zeros(size(I1)/levels(1));
v=zeros(size(I2)/levels(1));

for scale = levels %multi scale
    
    I1_ = imresize(I1, size(I1)/scale, 'bilinear');
    I2_ = imresize(I2, size(I2)/scale, 'bilinear');
    
    if scale == levels(1)
        iter = 300 ; 
        talyor = 5 ; 
    end
    
    if scale == levels(2)
        iter = 200 ; 
        talyor = 4 ; 
    end
    
    if scale == levels(3)
        iter = 100 ; 
        talyor = 3 ; 
    end
    
    if scale == levels(4)
        iter = 100 ; 
        talyor = 2 ; 
    end
    
    for expension = 1 : talyor
        
        u0 = u; 
        v0 = v;
        I1_warped = warpTarget(I1_, u0, v0);
        [u, v] = tvl1_optimizer(u, v, u0, v0, I1_warped, I2_, iter);
        
    end
    
    if scale ~= levels(end) 
        u = imresize(u, 2, 'bilinear');
        v = imresize(v, 2, 'bilinear');
    end

end

% [m,n]=size(I1);
% D=zeros(m,n,2);
% D(:,:,1) = u;
% D(:,:,2) = v;
% warped_im1 = imwarp(im1,D); 
% figure; imshow(warped_im1); title('warped frame 1, which should be close to frame 2');
% figure; imshow(im2); title('frame 2');
% figure; imshowpair(warped_im1(2:end-1,2:end-1),im2(2:end-1,2:end-1)); title('after optical flow');
% figure; imshow(flowToColor(D(2:end-1,2:end-1,:))); title('optical flow');

mag = sqrt(u.^2 + v.^2);
an = (atan2(v,u)+pi)/(2*pi);
figure; imshow(hsv2rgb(an,mag,mag)); axis off; axis equal;

function warp_I1 = warpTarget(I1_, u0, v0)
[m,n]=size(I1_);
D = zeros(m,n,2);
D(:,:,1) = u0;
D(:,:,2) = v0;
warp_I1 = imwarp(I1_,D);

