function show_deformed_grid_direct(source, u0, v0, padNum, stepsize, color, t)

[m,n] = size(source);
[Y, X]=meshgrid(1:n,1:m);   
Deform_Y = u0 + Y ;
Deform_X = v0 + X ;

figure; imagesc(source(padNum+1:m-padNum,padNum+1:n-padNum)-source(padNum+1:m-padNum,padNum+1:n-padNum)+1); colormap(gray); title('Grid'); axis off; axis equal;
hold on;

Deform_X = Deform_X(padNum+1:m-padNum,padNum+1:n-padNum) - padNum;
Deform_Y = Deform_Y(padNum+1:m-padNum,padNum+1:n-padNum) - padNum;

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

% saveas(gcf,[num2str(t),'grid.png'])