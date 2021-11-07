%图像的灰度处理以及二维离散傅里叶变换
%ZihanGan 2021.9.17

% 读取图像
%fullname = get_full_filename('hust.png');
image = imread('hust.jpg');
shape = size(image);
%灰度处理,对三通道进行归一化后叠加
%image=double(image)/255;
%gray=sum(image,3)/3;
gray=rgb2gray(image);
gray=double(gray);
%imshow(gray,[]);
%中心平移
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
%F=fft2(gray);
%幅度谱中心平移
%F=fftshift(F);
%取幅值
F=abs(F);
%imshow(F,[]);
%取对数
F=log(F);
imshow(F,[]);
x=[2.56	2.44	2.4	2.32	2.22	2.1	2.02	1.96	1.91	1.88	1.84	1.76	1.72	1.68	1.6	1.56	1.5	1.46	1.42	1.34	1.3	1.26	1.22	1.14	1.08]
