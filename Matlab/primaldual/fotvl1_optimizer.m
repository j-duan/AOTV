function [x1_k, x2_k] = fotvl1_optimizer(u0,v0,im1,im2,maxIter,lambda,tol)

tau   = 1/sqrt(4096)/lambda;
sigma = 1/sqrt(4096)/lambda;
theta = 1;

[Ix, Iy] = computeDerivatives(im1);
It = im1 - im2;

J11 = Ix.*Ix;
J22 = Iy.*Iy;

[m,n]=size(im1);
y11_k=zeros(m,n);  y12_k=zeros(m,n); y13_k=zeros(m,n); y14_k=zeros(m,n); y15_k=zeros(m,n);
y21_k=zeros(m,n);  y22_k=zeros(m,n); y23_k=zeros(m,n); y24_k=zeros(m,n); y25_k=zeros(m,n);
x1_k =zeros(m,n);   x2_k =zeros(m,n);
x1ba_k =zeros(m,n); x2ba_k =zeros(m,n);


tStart = tic;
for i = 1 : maxIter

    temp_u = x1_k;
    temp_v = x2_k;   
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % using L1 data term ending up with a thresholding equation
    tmp11 = y11_k + lambda*sigma*Bx(Fx(Bx(Fx(x1ba_k))));
    tmp12 = y12_k + lambda*sigma*Fx(Bx(Fx(Fy(x1ba_k))));
    tmp13 = y13_k + lambda*sigma*Bx(Fx(By(Fy(x1ba_k))));
    tmp14 = y14_k + lambda*sigma*Fx(Fy(By(Fy(x1ba_k))));
    tmp15 = y15_k + lambda*sigma*By(Fy(By(Fy(x1ba_k))));
    tmp21 = y21_k + lambda*sigma*Bx(Fx(Bx(Fx(x2ba_k))));
    tmp22 = y22_k + lambda*sigma*Fx(Bx(Fx(Fy(x2ba_k))));
    tmp23 = y23_k + lambda*sigma*Bx(Fx(By(Fy(x2ba_k))));
    tmp24 = y24_k + lambda*sigma*Fx(Fy(By(Fy(x2ba_k))));
    tmp25 = y25_k + lambda*sigma*By(Fy(By(Fy(x2ba_k))));
    tmp = max(sqrt(tmp11.^2+4*tmp12.^2+6*tmp13.^2+4*tmp14.^2+tmp15.^2+tmp21.^2+4*tmp22.^2+6*tmp23.^2+4*tmp24.^2+tmp25.^2),1);
    y11_k1 = tmp11./tmp;
    y12_k1 = tmp12./tmp;
    y13_k1 = tmp13./tmp;
    y14_k1 = tmp14./tmp;
    y15_k1 = tmp15./tmp;
    y21_k1 = tmp21./tmp;
    y22_k1 = tmp22./tmp;
    y23_k1 = tmp23./tmp;
    y24_k1 = tmp24./tmp;
    y25_k1 = tmp25./tmp;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tmp1 = x1_k - lambda*tau*div(y11_k1, y12_k1, y13_k1, y14_k1, y15_k1);
    tmp2 = x2_k - lambda*tau*div(y21_k1, y22_k1, y23_k1, y24_k1, y25_k1);
    [x1_k1, x2_k1] = resolvent_operator(tmp1, tmp2, u0, v0, Ix, Iy, It, J11, J22, 1/tau);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x1ba_k1 = x1_k1 + theta*(x1_k1 - x1_k);
    x2ba_k1 = x2_k1 + theta*(x2_k1 - x2_k); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x1_k    = x1_k1;
    x2_k    = x2_k1;
    x1ba_k  = x1ba_k1;
    x2ba_k  = x2ba_k1;
    y11_k   = y11_k1;
    y12_k   = y12_k1;
    y13_k   = y13_k1;
    y14_k   = y14_k1;
    y15_k   = y15_k1;
    y21_k   = y21_k1;
    y22_k   = y22_k1;
    y23_k   = y23_k1;
    y24_k   = y24_k1;
    y25_k   = y25_k1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % convergnece check
    stop1 = sum(sum(abs(x1_k-temp_u)))/(sum(sum(abs(temp_u)))+eps);
    stop2 = sum(sum(abs(x2_k-temp_v)))/(sum(sum(abs(temp_v)))+eps);
    stop(i) = max(stop1, stop2);
    time(i) = toc(tStart);
    if stop(i) < tol
        if i > 2
            fprintf('    iterate %d times, stop due to converge to tolerance \n', i);
            break; % set break crmaxIterion
        end
    end
end
if i == maxIter
    fprintf('   iterate %d times, stop due to reach to max iteration \n', i);
end

% Compute divergence using backward derivative
function f = div(a,b,c,d,e)
f = Bx(Fx(Bx(Fx(a))))+4*By(Bx(Fx(Bx(b))))+6*By(Fy(Bx(Fx(c))))+4*By(Fy(By(Bx(d))))+By(Fy(By(Fy(e))));

% Forward derivative operator on x with boundary condition u(:,:,1)=u(:,:,1)
function Fxu = Fx(u)
[m,n] = size(u);
Fxu = circshift(u,[0 -1])-u;
Fxu(:,n) = zeros(m,1);

% Forward derivative operator on y with boundary condition u(1,:,:)=u(m,:,:)
function Fyu = Fy(u)
[m,n] = size(u);
Fyu = circshift(u,[-1 0])-u;
Fyu(m,:) = zeros(1,n);

% Backward derivative operator on x with boundary condition Bxu(:,1)=u(:,1)
function Bxu = Bx(u)
[~,n] = size(u);
Bxu = u - circshift(u,[0 1]);
Bxu(:,1) = u(:,1);
Bxu(:,n) = -u(:,n-1);

% Backward derivative operator on y with boundary condition Bxu(1,:)=u(1,:)
function Byu = By(u)
[m,~] = size(u);
Byu = u - circshift(u,[1 0]);
Byu(1,:) = u(1,:);
Byu(m,:) = -u(m-1,:);

function [ux, uy]=computeDerivatives(u)
[m,n]=size(u);
C1 = circshift(u,[0 -1]); C1(:,n) = C1(:,n-1);
C2 = circshift(u,[0 1]);  C2(:,1) = C2(:,2);
C3 = circshift(u,[-1 0]); C3(m,:) = C3(m-1,:);
C4 = circshift(u,[1 0]);  C4(1,:) = C4(2,:);
ux=(C1-C2)/2;
uy=(C3-C4)/2;

function [u, v] = resolvent_operator(tmp1, tmp2, u0, v0, Ix, Iy, It, J11, J22, theta_w)
[m,n]=size(tmp1);
mask_1 = zeros(m,n);
mask_2 = zeros(m,n); 
mask_3 = zeros(m,n);
rho_w = Ix .* (tmp1-u0) + Iy .* (tmp2-v0) + It;
mask_1(rho_w < - ( J11 + J22 ) / theta_w) = 1;
mask_2(rho_w >   ( J11 + J22 ) / theta_w) = 1;
mask_3(-( J11 + J22 ) / theta_w <= rho_w & rho_w <= ( J11 + J22 ) / theta_w) = 1;
u = tmp1 + (Ix / theta_w).*mask_1 + (-Ix / theta_w).*mask_2 + (-rho_w .* Ix ./ ( J11 + J22 + eps )).*mask_3;
v = tmp2 + (Iy / theta_w).*mask_1 + (-Iy / theta_w).*mask_2 + (-rho_w .* Iy ./ ( J11 + J22 + eps )).*mask_3;