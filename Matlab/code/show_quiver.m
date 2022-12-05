function show_quiver(source, u0, v0, padNum, t)


[m,n] = size(source);

figure; imagesc(source(padNum+1:m-padNum,padNum+1:n-padNum)); colormap(gray); title('DVF'); axis off; axis equal;
hold on;
opflow = opticalFlow(u0(padNum+1:m-padNum,padNum+1:n-padNum),v0(padNum+1:m-padNum,padNum+1:n-padNum));
plot(opflow, 'DecimationFactor',[3 3],'ScaleFactor',1);
q = findobj(gca,'type','Quiver');
q.Color = 'r';
q.LineWidth = 1;
saveas(gcf,[num2str(t),'DVF.png'])


% [Y, X]=meshgrid(1:n,1:m);   
% Deform_Y = u0 + Y ;
% Deform_X = v0 + X ;
% 
% hold on
% stepsize = 3;
% color = 'g';
% Deform_X = Deform_X(padNum+1:m-padNum,padNum+1:n-padNum) - padNum;
% Deform_Y = Deform_Y(padNum+1:m-padNum,padNum+1:n-padNum) - padNum;
% 
% %plot the vertical lines
% for i = 1 : stepsize : size(Deform_X,1)
%     plot(Deform_Y(i,:), Deform_X(i,:), 'Color', color);
%     hold on
% end
% 
% %plot the horizontal lines
% for j = 1 : stepsize : size(Deform_X,2)
%     plot(Deform_Y(:,j), Deform_X(:,j), 'Color', color);
%     hold on
% end
% 
% set(gca, 'YDir','reverse')
% axis image
% axis off
% hold off

saveas(gcf,[num2str(t),'dvf_grid.png'])