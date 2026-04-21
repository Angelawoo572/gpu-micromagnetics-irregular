clear
a=load('outb03');
nn=a(1,1);
k=1;
%a(2,2)=0.0;
nn2=nn/2;
for jy=1:nn
    for ix=1:nn
        k=k+1;    
        mx(jy,ix)=a(k,1);
        my(jy,ix)=a(k,2);
        mz(jy,ix)=a(k,3);
        x(jy,ix)=ix;
        y(jy,ix)=jy;
    end
end
figure(1)
quiver(x,y,mx,my);
