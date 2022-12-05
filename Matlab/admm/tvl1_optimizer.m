function [u,v] = tvl1_optimizer(u0,v0,im1,im2,maxIter,lambda,tol)

theta_1 = 1;   % convergnece parameter
theta_2 = 0.1; % convergnece parameter
alpha   = 1.8; % relaxation parameter

[Ix, Iy] = computeDerivatives(im1);
It = im1 - im2;
J11 = Ix.*Ix;
J22 = Iy.*Iy;

[m,n]=size(im1);
b11=zeros(m,n); b12=zeros(m,n);
b21=zeros(m,n); b22=zeros(m,n);
d1 =zeros(m,n); d2 =zeros(m,n);
w11=zeros(m,n); w12=zeros(m,n);
w21=zeros(m,n); w22=zeros(m,n);
Wx =zeros(m,n); Wy =zeros(m,n);
u =zeros(m,n);  v =zeros(m,n);

[Y,X]=meshgrid(0:n-1,0:m-1);
% G=cos(pi*X/m)+cos(pi*Y/n)-2;
G = -2*(cos(pi*X/m)+cos(pi*Y/n)-2);
% G=(cos(2*pi*X/m)+cos(2*pi*Y/n)-2);
order = 1;

tStart = tic;
for i=1 : maxIter
    
    temp_u = u;
    temp_v = v;   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [u, v] = resolvent_operator(Wx-d1, Wy-d2, u0, v0, Ix, Iy, It, J11, J22, theta_2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    u = alpha * u + (1-alpha) * Wx;
    v = alpha * v + (1-alpha) * Wy;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update u1 using dct
    div_w_b=Bx(w11-b11)+By(w12-b12);
    g=theta_2*(u+d1)-theta_1*div_w_b;
    Wx=(mirt_idctn(mirt_dctn(g)./(theta_2+theta_1*G.^order)));

    % update u2 using dct
    div_w_b=Bx(w21-b21)+By(w22-b22);
    g=theta_2*(v+d2)-theta_1*div_w_b;
    Wy=(mirt_idctn(mirt_dctn(g)./(theta_2+theta_1*G.^order)));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    c11=Fx(Wx)+b11;
    c12=Fy(Wx)+b12;
    c21=Fx(Wy)+b21;
    c22=Fy(Wy)+b22;
    
    abs_c=sqrt(c11.^2+c12.^2+c21.^2+c22.^2+eps);
    w11=max(abs_c-lambda/theta_1,0).*c11./abs_c;
    w12=max(abs_c-lambda/theta_1,0).*c12./abs_c;
    w21=max(abs_c-lambda/theta_1,0).*c21./abs_c;
    w22=max(abs_c-lambda/theta_1,0).*c22./abs_c;
         
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update Bregman iterative parameters b
    b11=c11-w11;
    b12=c12-w12;
    b21=c21-w21;
    b22=c22-w22;
    d1=d1+u-Wx;
    d2=d2+v-Wy;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % convergnece check
    stop1 = sum(sum(abs(u-temp_u)))/(sum(sum(abs(temp_u)))+eps);
    stop2 = sum(sum(abs(v-temp_v)))/(sum(sum(abs(temp_v)))+eps);
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

% function [u, v] = resolvent_operator(tmp1, tmp2, u0, v0, Ix, Iy, It, J11, J22, theta_2)
% [m,n]=size(tmp1);
% mask_1 = zeros(m,n);
% mask_2 = zeros(m,n); 
% mask_3 = zeros(m,n);
% rho_w = Ix .* (tmp1-u0) + Iy .* (tmp2-v0) + It;
% mask_1(rho_w < - ( J11 + J22 ) / theta_2) = 1;
% mask_2(rho_w >   ( J11 + J22 ) / theta_2) = 1;
% mask_3(-( J11 + J22 ) / theta_2 <= rho_w & rho_w <= ( J11 + J22 ) / theta_2) = 1;
% u = tmp1 + (Ix / theta_2).*mask_1 + (-Ix / theta_2).*mask_2 + (-rho_w .* Ix ./ ( J11 + J22 + eps )).*mask_3;
% v = tmp2 + (Iy / theta_2).*mask_1 + (-Iy / theta_2).*mask_2 + (-rho_w .* Iy ./ ( J11 + J22 + eps )).*mask_3;

function [u, v] = resolvent_operator(tmp1, tmp2, u0, v0, Ix, Iy, It, J11, J22, theta_w)
rho_w = Ix .* (tmp1-u0) + Iy .* (tmp2-v0) + It;
zhat = theta_w * rho_w ./ (J11 + J22 + eps);
tmp = zhat./max(abs(zhat), 1);
u = tmp1 - (Ix / theta_w).*tmp;
v = tmp2 - (Iy / theta_w).*tmp;