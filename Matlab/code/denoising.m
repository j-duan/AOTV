function denoising()
% close all

f0 = imread('castleNoise.bmp');

f0 = double(f0(:,:,1));

f = f0;
figure; imagesc(f); colormap(gray); axis off; axis equal;

t = 1e-3;
lambda = 1;
sigma = sqrt(2*t);
u0 = f0;
for i = 1 : 10
    
    u = GaussianReplicateBoundary(u0,sigma);
    
    u0 = (t*lambda*f0 + u)/(1+t*lambda);
    
end
figure; imagesc(u); colormap(gray); axis off; axis equal;