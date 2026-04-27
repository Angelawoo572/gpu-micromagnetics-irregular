%% 
% clear;
% clear all;
% fn_main=strcat('output.txt');
% ff=strcat(fn_main);
% aa=readmatrix(fn_main);
% [nt,tmp]=size(aa);
% nx = aa(1,2)/3;
% ny = aa(1,3);
% nqs=nx*ny;
% nframe=nt/(nqs+1);
% j=1;
% figure(1)
% hold off;
% mt=0;
% nframe_end=floor(nframe);
% for iframe=1:nframe_end
%     jt=iframe
%     mt=mt+1;
%     for iy=1:ny
%         for ix=1:nx
%             mt=mt+1;
%             xp(iy,ix)=ix;
%             yp(iy,ix)=iy;
%             zp(iy,ix)=0;
%             mx(iy,ix)=aa(mt,1);
%             my(iy,ix)=aa(mt,2);
%             mz(iy,ix)=aa(mt,3);
%         end
%     end  
%     figure(1)
%     view(3);
%     %pcolor(mx);
%     %shading interp;
%     %hold on
%     grid off;
%     quiver3(xp,yp,zp,mx,my,mz,0.7,'color','red');
%     %view(3)
%     axis off
%     axis image
%     %if jt < 10
%     %    fnp=sprintf('00%d.jpg',jt);
%     %elseif jt < 100
%     %    fnp=sprintf('0%d.jpg',jt);
%     %else
%     %    fnp=sprintf('%d.jpg',jt);
%     %end
%     %str_jpg=strcat(fn_main,fnp);
%     %str_jpg = sprintf('%03d.jpg',jt); % jpeg file name    
%     %saveas(gcf,str_jpg); % save the graph to a jpeg file
%     %hold off
%     %figure(2)
%     %mesh(mz);
% end

% clear; clc; close all;
% 
% fn = 'output.txt';
% aa = readmatrix(fn);
% [nt, ~] = size(aa);
% 
% scalar_nx = aa(1,2);
% ny        = aa(1,3);
% nx        = scalar_nx / 3;   % physical x cells
% nqs       = nx * ny;
% frame_len = nqs + 1;
% nframe    = floor(nt / frame_len);
% 
% stride = 4;   % 可改成 2 / 4 / 8
% xs = 1:stride:nx;
% ys = 1:stride:ny;
% [X,Y] = meshgrid(xs, ys);
% Z = zeros(size(X));
% 
% gif_name = 'quiver_animation.gif';
% 
% figure('Color','w');
% for iframe = 1:nframe
%     header_row = 1 + (iframe-1)*frame_len;
%     data_start = header_row + 1;
%     data_end   = header_row + nqs;
% 
%     if data_end > nt
%         fprintf('Frame %d incomplete, stopping.\n', iframe);
%         break;
%     end
% 
%     tnow  = aa(header_row, 1);
%     block = aa(data_start:data_end, :);
% 
%     MX_full = reshape(block(:,1), [nx, ny])';
%     MY_full = reshape(block(:,2), [nx, ny])';
%     MZ_full = reshape(block(:,3), [nx, ny])';
% 
%     MX = MX_full(ys, xs);
%     MY = MY_full(ys, xs);
%     MZ = MZ_full(ys, xs);
%     cla;
% 
%     quiver3(X, Y, Z, MX, MY, MZ, 0.7, 'r');
%     axis image;
%     axis off;
%     view(3);
%     title(sprintf('Magnetization, frame %d / %d, t = %.4f', iframe, nframe, tnow), ...
%           'FontSize', 14);
%     drawnow;
% 
%     frame = getframe(gcf);
%     im = frame2im(frame);
%     [A, map] = rgb2ind(im, 256);
% 
%     if iframe == 1
%         imwrite(A, map, gif_name, 'gif', 'LoopCount', Inf, 'DelayTime', 0.15);
%     else
%         imwrite(A, map, gif_name, 'gif', 'WriteMode', 'append', 'DelayTime', 0.15);
%     end
% end
% 
% fprintf('Saved GIF: %s\n', gif_name);

% clear; clc; close all;
% 
% fn = 'output.txt';
% aa = readmatrix(fn);
% 
% scalar_nx = aa(1,2);
% ny        = aa(1,3);
% nx        = scalar_nx / 3;
% nqs       = nx * ny;
% 
% block = aa(2:1+nqs, :);
% 
% MX = reshape(block(:,1), [nx, ny])';
% MY = reshape(block(:,2), [nx, ny])';
% MZ = reshape(block(:,3), [nx, ny])';
% 
% [x, y] = meshgrid(1:nx, 1:ny);
% 
% figure('Color','w');
% surf(x, y, MX);
% shading interp;
% colorbar;
% axis tight;
% xlabel('x');
% ylabel('y');
% zlabel('m_x');
% title('Final state: m_x');
% 
% % 如果想从上往下看 hole/纹理
% figure('Color','w');
% surf(x, y, MX);
% shading interp;
% view(2);
% axis equal tight;
% colorbar;
% xlabel('x');
% ylabel('y');
% title('Final state: m_x top view');

% clear; clc; close all;
% 
% fn = 'output.txt';
% aa = readmatrix(fn);
% 
% scalar_nx = aa(1,2);
% ny        = aa(1,3);
% nx        = scalar_nx / 3;
% nqs       = nx * ny;
% 
% block = aa(2:1+nqs, :);
% 
% MX = reshape(block(:,1), [nx, ny])';
% MY = reshape(block(:,2), [nx, ny])';
% MZ = reshape(block(:,3), [nx, ny])';
% 
% [x, y] = meshgrid(1:nx, 1:ny);
% 
% % map
% % mx(j,i) - mx(j-1,i)
% dMx_dy = MX - circshift(MX, [1, 0]);
% 
% % -( my(j,i) - my(j,i-1) )
% dMy_dx = MY - circshift(MY, [0, 1]);
% 
% % map = mx(j,i)-mx(j-1,i) - [my(j,i)-my(j,i-1)]
% MAP = dMx_dy - dMy_dx;
% 
% %
% MAP(1,:) = 0;
% MAP(:,1) = 0;
% 
% % ===== 原来的 mx top view =====
% figure('Color','w');
% surf(x, y, MX);
% shading interp;
% view(2);
% axis equal tight;
% colorbar;
% xlabel('x');
% ylabel('y');
% title('Final state: m_x top view');
% 
% % ===== 老师要看的 map =====
% figure('Color','w');
% surf(x, y, MAP);
% shading interp;
% view(2);
% axis equal tight;
% colorbar;
% xlabel('x');
% ylabel('y');
% title('map = [m_x(j,i)-m_x(j-1,i)] - [m_y(j,i)-m_y(j,i-1)]');
% 
% % ===== 3D map =====
% figure('Color','w');
% surf(x, y, MAP);
% shading interp;
% colorbar;
% axis tight;
% xlabel('x');
% ylabel('y');
% zlabel('map');
% title('DMI-like map');
% 
% clear; clc; close all;
% 
% fn = 'output.txt';
% aa = readmatrix(fn);
% [nt, ~] = size(aa);
% 
% scalar_nx = aa(1,2);
% ny        = aa(1,3);
% nx        = scalar_nx / 3;
% nqs       = nx * ny;
% 
% frame_len = nqs + 1;
% nframe    = floor(nt / frame_len);
% 
% [x, y] = meshgrid(1:nx, 1:ny);
% 
% gif_name = 'surf_map_animation.gif';
% 
% figure('Color','w');
% 
% for iframe = 1:nframe
%     header_row = 1 + (iframe-1)*frame_len;
%     data_start = header_row + 1;
%     data_end   = header_row + nqs;
% 
%     if data_end > nt
%         break;
%     end
% 
%     tnow  = aa(header_row, 1);
%     block = aa(data_start:data_end, :);
% 
%     MX = reshape(block(:,1), [nx, ny])';
%     MY = reshape(block(:,2), [nx, ny])';
%     MZ = reshape(block(:,3), [nx, ny])';
% 
%     % ===== 老师要看的 map =====
%     dMx_dy = MX - circshift(MX, [1, 0]);
%     dMy_dx = MY - circshift(MY, [0, 1]);
%     MAP = dMx_dy - dMy_dx;
% 
%     % 去掉 circshift 造成的边界假信号
%     MAP(1,:) = 0;
%     MAP(:,1) = 0;
% 
%     cla;
% 
%     % ===== surf 动画 =====
%     surf(x, y, MAP);
%     shading interp;
%     view(2);              % top view。如果想 3D 就改成 view(3)
%     axis equal tight;
%     colorbar;
%     xlabel('x');
%     ylabel('y');
%     title(sprintf('DMI-like map, frame %d/%d, t = %.4f', ...
%           iframe, nframe, tnow));
% 
%     drawnow;
% 
%     frame = getframe(gcf);
%     im = frame2im(frame);
%     [A, map] = rgb2ind(im, 256);
% 
%     if iframe == 1
%         imwrite(A, map, gif_name, 'gif', ...
%             'LoopCount', Inf, 'DelayTime', 0.15);
%     else
%         imwrite(A, map, gif_name, 'gif', ...
%             'WriteMode', 'append', 'DelayTime', 0.15);
%     end
% end
% 
% fprintf('Saved GIF: %s\n', gif_name);

% clear; clc; close all;
% 
% fn = 'output_i4.txt';
% aa = readmatrix(fn);
% [nt, ~] = size(aa);
% 
% scalar_nx = aa(1,2);
% ny        = aa(1,3);
% nx        = scalar_nx / 3;
% nqs       = nx * ny;
% frame_len = nqs + 1;
% nframe    = floor(nt / frame_len);
% 
% stride = 2;   % 可以调小一点更细腻
% xs = 1:stride:nx;
% ys = 1:stride:ny;
% [X,Y] = meshgrid(xs, ys);
% 
% gif_name = 'topview_surf.gif';
% 
% figure('Color','w');
% 
% for iframe = 1:nframe
%     header_row = 1 + (iframe-1)*frame_len;
%     data_start = header_row + 1;
%     data_end   = header_row + nqs;
% 
%     if data_end > nt
%         break;
%     end
% 
%     tnow  = aa(header_row, 1);
%     block = aa(data_start:data_end, :);
% 
%     MX_full = reshape(block(:,1), [nx, ny])';
%     MX = MX_full(ys, xs);
% 
%     cla;
% 
%     % ===== TOP VIEW =====
%     surf(X, Y, MX);
% 
%     shading interp;   % 平滑颜色（关键）
%     view(2);
%     axis equal tight;
%     colorbar;
% 
%     xlabel('x');
%     ylabel('y');
% 
%     title(sprintf('m_x (top view), frame %d/%d, t=%.3f', ...
%         iframe, nframe, tnow));
% 
%     drawnow;
% 
%     frame = getframe(gcf);
%     im = frame2im(frame);
%     [A, cmap] = rgb2ind(im, 256);
% 
%     if iframe == 1
%         imwrite(A, cmap, gif_name, 'gif', ...
%             'LoopCount', Inf, 'DelayTime', 0.1);
%     else
%         imwrite(A, cmap, gif_name, 'gif', ...
%             'WriteMode', 'append', 'DelayTime', 0.1);
%     end
% end
% 
% fprintf('Saved GIF: %s\n', gif_name);

clear; clc; close all;

fn = 'output.txt';
aa = readmatrix(fn);
[nt, ~] = size(aa);

scalar_nx = aa(1,2);
ny        = aa(1,3);
nx        = scalar_nx / 3;
nqs       = nx * ny;
frame_len = nqs + 1;
nframe    = floor(nt / frame_len);

stride = 2;
xs = 1:stride:nx;
ys = 1:stride:ny;
[X,Y] = meshgrid(xs, ys);

gif_top  = 'mx_topview.gif';
gif_3d   = 'mx_3d_surf.gif';
gif_map  = 'dmi_map_topview.gif';

for iframe = 1:nframe
    header_row = 1 + (iframe-1)*frame_len;
    data_start = header_row + 1;
    data_end   = header_row + nqs;

    if data_end > nt
        break;
    end

    tnow  = aa(header_row, 1);
    block = aa(data_start:data_end, :);

    MX_full = reshape(block(:,1), [nx, ny])';
    MY_full = reshape(block(:,2), [nx, ny])';

    MX = MX_full(ys, xs);
    MY = MY_full(ys, xs);

    % 老师写的 map:
    % mx(j,i)-mx(j-1,i) - [my(j,i)-my(j,i-1)]
    dMx_dy = MX_full - circshift(MX_full, [1, 0]);
    dMy_dx = MY_full - circshift(MY_full, [0, 1]);
    MAP_full = dMx_dy - dMy_dx;
    MAP_full(1,:) = 0;
    MAP_full(:,1) = 0;
    MAP = MAP_full(ys, xs);

    % ========== 1. mx top view ==========
    fig1 = figure(1); clf;
    set(fig1, 'Color', 'w');
    surf(X, Y, MX);
    shading interp;
    view(2);
    axis equal tight;
    colorbar;
    xlabel('x'); ylabel('y');
    title(sprintf('m_x top view, frame %d/%d, t=%.3f', iframe, nframe, tnow));
    drawnow;
    write_gif_frame(fig1, gif_top, iframe, 0.10);

    % ========== 2. mx 3D surf ==========
    fig2 = figure(2); clf;
    set(fig2, 'Color', 'w');
    surf(X, Y, MX);
    shading interp;
    view(3);
    axis tight;
    colorbar;
    xlabel('x'); ylabel('y'); zlabel('m_x');
    title(sprintf('m_x 3D surf, frame %d/%d, t=%.3f', iframe, nframe, tnow));
    drawnow;
    write_gif_frame(fig2, gif_3d, iframe, 0.10);

    % ========== 3. DMI-like map top view ==========
    fig3 = figure(3); clf;
    set(fig3, 'Color', 'w');
    surf(X, Y, MAP);
    shading interp;
    view(2);
    axis equal tight;
    colorbar;
    xlabel('x'); ylabel('y');
    title(sprintf('map = d_y m_x - d_x m_y, frame %d/%d, t=%.3f', iframe, nframe, tnow));
    drawnow;
    write_gif_frame(fig3, gif_map, iframe, 0.10);
end

fprintf('Saved GIFs:\n');
fprintf('  %s\n', gif_top);
fprintf('  %s\n', gif_3d);
fprintf('  %s\n', gif_map);

function write_gif_frame(fig, gif_name, iframe, delay)
    frame = getframe(fig);
    im = frame2im(frame);
    [A, cmap] = rgb2ind(im, 256);

    if iframe == 1
        imwrite(A, cmap, gif_name, 'gif', ...
            'LoopCount', Inf, 'DelayTime', delay);
    else
        imwrite(A, cmap, gif_name, 'gif', ...
            'WriteMode', 'append', 'DelayTime', delay);
    end
end