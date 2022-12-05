function [u,v] = H1_optimizer(u0,v0,im1,im2,maxIter,lambda,tol)

theta_w = 0.01; % convergnece parameter

[Ix, Iy] = computeDerivatives(im1);
It = im1 - im2;

J11 = Ix.*Ix;
J22 = Iy.*Iy;
J12 = Ix.*Iy;
J21 = Iy.*Ix;
J13 = Ix.*(It - Ix.*u0 - Iy.*v0);
J23 = Iy.*(It - Ix.*u0 - Iy.*v0);

[m,n]=size(im1);
d1 =zeros(m,n);
d2 =zeros(m,n);
Wx =zeros(m,n);
Wy =zeros(m,n);
u =zeros(m,n);
v =zeros(m,n);

[Y,X]=meshgrid(0:n-1,0:m-1);
% G=2*(cos(pi*X/m)+cos(pi*Y/n)-2);
G=2*(cos(2*pi*X/m)+cos(2*pi*Y/n)-2);

for i  = 1 : maxIter
    
    temp_u = u;
    temp_v = v;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % using L2 data term
    %     h1 = theta_w * (Wx-d1) - J13;
    %     h2 = theta_w * (Wy-d2) - J23;
    %     u = ( (J22 + theta_w).*h1 - J12.*h2) ./ ((J11 + theta_w).*(J22 + theta_w) - J12.*J21);
    %     v = ( (J11 + theta_w).*h2 - J21.*h1) ./ ((J11 + theta_w).*(J22 + theta_w) - J12.*J21);
    
    % using L1 data term
    %     rho_w = Ix .* (Wx-d1) + Iy .* (Wy-d2) + (It - Ix.*u0 - Iy.*v0);
    %     mask_1=zeros(m,n);
    %     mask_2=zeros(m,n);
    %     mask_3=zeros(m,n);
    %     mask_1(rho_w < - ( J11 + J22 ) / theta_w) = 1;
    %     mask_2(rho_w >   ( J11 + J22 ) / theta_w) = 1;
    %     mask_3(-( J11 + J22 ) / theta_w <= rho_w & rho_w <= ( J11 + J22 ) / theta_w) = 1;
    %     u = Wx-d1 + (Ix / theta_w).*mask_1 + (-Ix / theta_w).*mask_2 + (-rho_w .* Ix ./ ( J11 + J22 + eps )).*mask_3;
    %     v = Wy-d2 + (Iy / theta_w).*mask_1 + (-Iy / theta_w).*mask_2 + (-rho_w .* Iy ./ ( J11 + J22 + eps )).*mask_3;
      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update U using dct
    u = ( (J22 + theta_w).*(Wx-d1) - J12.*(Wy-d2) - J13) ./ (J11 + J22 + theta_w);
    v = ( (J11 + theta_w).*(Wy-d2) - J21.*(Wx-d1) - J23) ./ (J11 + J22 + theta_w);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    u = alpha * u + (1-alpha) * Wx;
    v = alpha * v + (1-alpha) * Wy;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update W using dct
    % Wx=(idct2(dct2(theta_w*(u+d1))./(theta_w+lambda*G.^2)));
    % Wy=(idct2(dct2(theta_w*(v+d2))./(theta_w+lambda*G.^2)));
    Wx=real(ifft2(fft2(theta_w*(u+d1))./(theta_w+lambda*G.^2)));
    Wy=real(ifft2(fft2(theta_w*(v+d2))./(theta_w+lambda*G.^2)));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update bregman parameters
    d1=d1+u-Wx;
    d2=d2+v-Wy;
    
    
    % convergnece check
    stop1 = sum(sum(abs(u-temp_u)))/(sum(sum(abs(temp_u)))+eps);
    stop2 = sum(sum(abs(v-temp_v)))/(sum(sum(abs(temp_v)))+eps);
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

function [ux, uy]=computeDerivatives(u)
[m,n]=size(u);
C1 = circshift(u,[0 -1]); C1(:,n) = C1(:,n-1);
C2 = circshift(u,[0 1]);  C2(:,1) = C2(:,2);
C3 = circshift(u,[-1 0]); C3(m,:) = C3(m-1,:);
C4 = circshift(u,[1 0]);  C4(1,:) = C4(2,:);
ux=(C1-C2)/2;
uy=(C3-C4)/2;