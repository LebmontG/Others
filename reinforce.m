%灰度化，计算灰度直方图和离散傅里叶变换频谱幅度图
%直方图均衡化和同态滤波操作
%ZihanGan 2021.11.1

% 读取图像
image = imread('org.jpg');
%imshow(image);
shape = size(image);
%灰度处理,对三通道进行归一化后叠加
%也可以直接调用rbg2gray，仅仅在叠加时三通道的权重有变化
%image=double(image)/255;
%gray=sum(image,3)/3;
gray=rgb2gray(image);
gray=double(gray);
%imshow(gray,[]);
%中心平移，实现傅里叶变换的中心移动
for x=1:shape(1)
    for y=1:shape(2)
        gray(x,y)=gray(x,y)*((-1)^(x+y));
    end
end
%二维离散傅里叶
Fux=zeros(shape(1),shape(2));
F=zeros(shape(1),shape(2));
[u,v]=meshgrid(0:shape(2)-1,0:shape(1)-1);
for x=1:shape(2)
    s=gray.*exp(-1j*2*(x-1)*pi*u./shape(2));
    Fux(:,x)=sum(s,2);
end
for y=1:shape(1)
    s=Fux.*exp(-1j*2*(y-1)*pi*v./shape(1));
    F(y,:)=sum(s,1);
end
%取幅值
F=abs(F);
%imshow(F,[]);
%取对数，如果没有对数操作，得到的频谱图几乎无法用人眼观察
F=log(F);
%输出频谱幅度图
imshow(F,[]);
%首先做浮点数变换，将区间映射到255，然后取整数
%也可以直接对原图（0，1）的像素统计分布
%gray=gray/max(max(gray))*255;
%uint8()可以将（0，1）的灰度图映射到（0，255）的灰度图
gray=uint8(gray);
%统计灰度元素直方图
tol=zeros(1,256);
for x=1:shape(1)
    for y=1:shape(2)
        tol(gray(x,y)+1)=tol(gray(x,y)+1)+1;
    end
end
%输出灰度直方图
bar(tol);
%求出每个像素的概率
tol=tol/shape(1)/shape(2);
%求概率分布
for x=2:256
    tol(x)=tol(x)+tol(x-1);
end
%bar(tol);
%直方图均衡化
%计算新图的灰度值
tol=int16(255*tol+0.5);
%可去除重复元素后观察
%tol=unique(tol);
%bar(tol);
%得到新的灰度图
mgi=zeros(shape(1),shape(2));
%灰度映射
for x=1:shape(1)
    for y=1:shape(2)
        mgi(x,y)=tol(gray(x,y)+1);
    end
end
%显示图像，这里得到了直方图均衡化的结果
imshow(mgi,[]);
%取对数
gray=rgb2gray(image);
gray=log(im2double(gray)+1e-10);
%二维离散傅里叶变化到频域操作
%这里由于离散傅里叶变换在得到频谱图的时候
%已经自主实现，为了保持代码的清晰与简洁不再重复
F=fftshift(fft2(gray));
%取形状
shape=size(F);
%imshow(log(abs(F)),[]);
%生成点阵
[x,y] = meshgrid(-shape(1)/2:shape(1)/2,-shape(2)/2:shape(2)/2);
%选择倒高斯滤波器，一个常用的频域滤波器
H = 0.25*(1-exp(-1*(x.^2+y.^2)/200))+0.25;
%滤波器增广，需要修正形状
H = imresize(H,[shape(1),shape(2)]);
%将滤波器与频谱图点乘，可以减少复杂度
F = F.*H;
%傅里叶逆变换，同样，以上已经实现，这里不再重复
F=real(ifft2(ifftshift(F)));
%取指数，即同态滤波的最后一步
F=exp(F);
%输出同态滤波图像
imshow(F);