function [u,v] = totvl1_optimizer(u0,v0,im1,im2,maxIter,lambda,tol)


theta_1 = 100; % convergnece parameter
theta_2 = 1; % convergnece parameter
alpha = 1.8;

[Ix, Iy] = computeDerivatives(im1);
It = im1 - im2;

J11 = Ix.*Ix;
J22 = Iy.*Iy;

[m,n]=size(im1);
b11=zeros(m,n); b12=zeros(m,n); b13=zeros(m,n); b14=zeros(m,n);
b21=zeros(m,n); b22=zeros(m,n); b23=zeros(m,n); b24=zeros(m,n);
w11=zeros(m,n); w12=zeros(m,n); w13=zeros(m,n); w14=zeros(m,n);
w21=zeros(m,n); w22=zeros(m,n); w23=zeros(m,n); w24=zeros(m,n);
d1 =zeros(m,n); d2 =zeros(m,n);
Wx =zeros(m,n); Wy =zeros(m,n);
u =zeros(m,n);  v =zeros(m,n);

[Y,X]=meshgrid(0:n-1,0:m-1);
G=cos(pi*X/m)+cos(pi*Y/n)-2;

tStart = tic;
for i = 1 : maxIter

    temp_u = u;
    temp_v = v;   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % using L2 data term
    %     J12 = Ix.*Iy;
    %     J21 = Iy.*Ix;    
    %     J13 = Ix.*(It - Ix.*u0 - Iy.*v0);
    %     J23 = Iy.*(It - Ix.*u0 - Iy.*v0);
    %     h1 = theta_2 * (Wx-d1) - J13;
    %     h2 = theta_2 * (Wy-d2) - J23;
    %     u = ( (J22 + theta_2).*h1 - J12.*h2) ./ ((J11 + theta_2).*(J22 + theta_2) - J12.*J21);
    %     v = ( (J11 + theta_2).*h2 - J21.*h1) ./ ((J11 + theta_2).*(J22 + theta_2) - J12.*J21);
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % using L1 data term ending up with a thresholding equation
    [u, v] = resolvent_operator(Wx-d1, Wy-d2, u0, v0, Ix, Iy, It, J11, J22, theta_2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    u = alpha * u + (1-alpha) * Wx;
    v = alpha * v + (1-alpha) * Wy;
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update u1 using DCT
    div_w_b =Bx(Fx(Bx(w11-b11)))+3*By(Bx(Fx(w12-b12)))+3*By(Bx(Fy(w13-b13)))+By(Fy(By(w14-b14)));
    g=theta_2*(u+d1)-theta_1*div_w_b;
    Wx=(mirt_idctn(mirt_dctn(g)./(theta_2-8*theta_1*G.^3)));
    
    % update u2 using DCT
    div_w_b =Bx(Fx(Bx(w21-b21)))+3*By(Bx(Fx(w22-b22)))+3*By(Bx(Fy(w23-b23)))+By(Fy(By(w24-b24)));
    g=theta_2*(v+d2)-theta_1*div_w_b;
    Wy=(mirt_idctn(mirt_dctn(g)./(theta_2-8*theta_1*G.^3)));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    % using L1 reg term ending up with annother thresholding equation
    c11=Fx(Bx(Fx(Wx)))+b11;
    c12=Bx(Fx(Fy(Wx)))+b12;
    c13=By(Fx(Fy(Wx)))+b13;
    c14=Fy(By(Fy(Wx)))+b14;
    
    c21=Fx(Bx(Fx(Wy)))+b21;
    c22=Bx(Fx(Fy(Wy)))+b22;
    c23=By(Fx(Fy(Wy)))+b23;
    c24=Fy(By(Fy(Wy)))+b24;
    
    abs_c=sqrt(c11.^2+3*c12.^2+3*c13.^2+c14.^2+c21.^2+3*c22.^2+3*c23.^2+c24.^2++eps);
    w11=max(abs_c-lambda/theta_1,0).*c11./abs_c;
    w12=max(abs_c-lambda/theta_1,0).*c12./abs_c;
    w13=max(abs_c-lambda/theta_1,0).*c13./abs_c;
    w14=max(abs_c-lambda/theta_1,0).*c14./abs_c;
    w21=max(abs_c-lambda/theta_1,0).*c21./abs_c;
    w22=max(abs_c-lambda/theta_1,0).*c22./abs_c;
    w23=max(abs_c-lambda/theta_1,0).*c23./abs_c;
    w24=max(abs_c-lambda/theta_1,0).*c24./abs_c;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update Bregman maxIterative parameters b
    b11=c11-w11;
    b12=c12-w12;
    b13=c13-w13;
    b14=c14-w14;
    b21=c21-w21;
    b22=c22-w22;
    b23=c23-w23;
    b24=c24-w24; 
    d1=d1+u-Wx;
    d2=d2+v-Wy;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % convergnece check
    stop1 = sum(sum(abs(u-temp_u)))/(sum(sum(abs(temp_u)))+eps);
    stop2 = sum(sum(abs(v-temp_v)))/(sum(sum(abs(temp_v)))+eps);
    stop(i) = max(stop1, stop2);
    time(i) = toc(tStart);
    if stop1 < tol && stop2 < tol
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

% central finite difference
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