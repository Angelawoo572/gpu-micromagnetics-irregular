clear;
clear all;
fn_main=strcat('dout');
ff=strcat(fn_main);
aa=load(fn_main);
[nt,tmp]=size(aa);
nx = aa(1,2)/3;
ny = aa(1,3);
nqs=nx*ny;
nframe=nt/(nqs+1);
j=1;
figure(1)
hold off;
mt=0;
nframe_end=floor(nframe);
for iframe=1:nframe_end
    jt=iframe
    mt=mt+1;
    for iy=1:ny
        for ix=1:nx
            mt=mt+1;
            xp(iy,ix)=ix;
            yp(iy,ix)=iy;
            zp(iy,ix)=0;
            mx(iy,ix)=aa(mt,1);
            my(iy,ix)=aa(mt,2);
            mz(iy,ix)=aa(mt,3);
        end
    end  
    figure(1)
    view(3);
    %pcolor(mx);
    %shading interp;
    %hold on
    grid off;
    quiver3(xp,yp,zp,mx,my,mz,0.7,'color','red');
    %view(3)
    axis off
    axis image
    %if jt < 10
    %    fnp=sprintf('00%d.jpg',jt);
    %elseif jt < 100
    %    fnp=sprintf('0%d.jpg',jt);
    %else
    %    fnp=sprintf('%d.jpg',jt);
    %end
    %str_jpg=strcat(fn_main,fnp);
    %str_jpg = sprintf('%03d.jpg',jt); % jpeg file name    
    %saveas(gcf,str_jpg); % save the graph to a jpeg file
    %hold off
    %figure(2)
    %mesh(mz);
end
