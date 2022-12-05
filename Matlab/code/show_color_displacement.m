function show_color_displacement(u0, v0, padNum, t)

[m,n] = size(u0);

figure; imagesc(flowToColor(u0(padNum+1:m-padNum,padNum+1:n-padNum),v0(padNum+1:m-padNum,padNum+1:n-padNum))); title('Displacement Field'); axis off; axis equal;

