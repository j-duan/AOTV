function plot_deformed_grid(img, u0, v0, stepsize, color)


[Y,X]=meshgrid(1:size(img,2),1:size(img,1));
Deform_X = X + v0 ;
Deform_Y = Y + u0 ;

%plot the vertical lines
for i = 1 : stepsize : size(Deform_X,1)
    plot(Deform_Y(i,:), Deform_X(i,:), 'Color', color);
    hold on
end

%plot the horizontal lines
for j = 1 : stepsize : size(Deform_X,2)
    plot(Deform_Y(:,j), Deform_X(:,j), 'Color', color);
    hold on
end

set(gca, 'YDir','reverse')
axis image
axis off
hold off














% [Y,X]=meshgrid(1:size(img,2), 1:size(img,1));
% Deform_X = X + v0 ;
% Deform_Y = Y + u0 ;
% 
% %plot the vertical lines
% for i = 1 : 3 : size(Deform_X,1)
%     line(Deform_Y(i,:),Deform_X(i,:),'Color','g');
%     set(gca, 'YDir','reverse')
%     hold on
% end
% 
% %plot the horizontal lines
% for j = 1 : 3 : size(Deform_X,2)
%     line(Deform_Y(:,j),Deform_X(:,j),'Color','g');
%     set(gca, 'YDir','reverse')
%     hold on
% end
% axis image
% title('Fwd. deformed grid')
% drawnow
% hold off