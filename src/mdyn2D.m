% %% 
% clear;
% clear all;
% fn_main=strcat('output_try2.txt');
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

clear; clc; close all;

fn = 'output.txt';
aa = readmatrix(fn);
[nt, ~] = size(aa);

scalar_nx = aa(1,2);
ny        = aa(1,3);
nx        = scalar_nx / 3;   % physical x cells
nqs       = nx * ny;
frame_len = nqs + 1;
nframe    = floor(nt / frame_len);

stride = 4;   % 可改成 2 / 4 / 8
xs = 1:stride:nx;
ys = 1:stride:ny;
[X,Y] = meshgrid(xs, ys);
Z = zeros(size(X));

gif_name = 'quiver_animation.gif';

figure('Color','w');
for iframe = 1:nframe
    header_row = 1 + (iframe-1)*frame_len;
    data_start = header_row + 1;
    data_end   = header_row + nqs;

    if data_end > nt
        fprintf('Frame %d incomplete, stopping.\n', iframe);
        break;
    end

    tnow  = aa(header_row, 1);
    block = aa(data_start:data_end, :);

    MX_full = reshape(block(:,1), [nx, ny])';
    MY_full = reshape(block(:,2), [nx, ny])';
    MZ_full = reshape(block(:,3), [nx, ny])';

    MX = MX_full(ys, xs);
    MY = MY_full(ys, xs);
    MZ = MZ_full(ys, xs);
    cla;

    quiver3(X, Y, Z, MX, MY, MZ, 0.7, 'r');
    axis image;
    axis off;
    view(3);
    title(sprintf('Magnetization, frame %d / %d, t = %.4f', iframe, nframe, tnow), ...
          'FontSize', 14);
    drawnow;

    frame = getframe(gcf);
    im = frame2im(frame);
    [A, map] = rgb2ind(im, 256);

    if iframe == 1
        imwrite(A, map, gif_name, 'gif', 'LoopCount', Inf, 'DelayTime', 0.15);
    else
        imwrite(A, map, gif_name, 'gif', 'WriteMode', 'append', 'DelayTime', 0.15);
    end
end

fprintf('Saved GIF: %s\n', gif_name);