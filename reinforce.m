%�ҶȻ�������Ҷ�ֱ��ͼ����ɢ����Ҷ�任Ƶ�׷���ͼ
%ֱ��ͼ���⻯��̬ͬ�˲�����
%ZihanGan 2021.11.1

% ��ȡͼ��
image = imread('org.jpg');
%imshow(image);
shape = size(image);
%�Ҷȴ���,����ͨ�����й�һ�������
%Ҳ����ֱ�ӵ���rbg2gray�������ڵ���ʱ��ͨ����Ȩ���б仯
%image=double(image)/255;
%gray=sum(image,3)/3;
gray=rgb2gray(image);
gray=double(gray);
%imshow(gray,[]);
%����ƽ�ƣ�ʵ�ָ���Ҷ�任�������ƶ�
for x=1:shape(1)
    for y=1:shape(2)
        gray(x,y)=gray(x,y)*((-1)^(x+y));
    end
end
%��ά��ɢ����Ҷ
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
%ȡ��ֵ
F=abs(F);
%imshow(F,[]);
%ȡ���������û�ж����������õ���Ƶ��ͼ�����޷������۹۲�
F=log(F);
%���Ƶ�׷���ͼ
imshow(F,[]);
%�������������任��������ӳ�䵽255��Ȼ��ȡ����
%Ҳ����ֱ�Ӷ�ԭͼ��0��1��������ͳ�Ʒֲ�
%gray=gray/max(max(gray))*255;
%uint8()���Խ���0��1���ĻҶ�ͼӳ�䵽��0��255���ĻҶ�ͼ
gray=uint8(gray);
%ͳ�ƻҶ�Ԫ��ֱ��ͼ
tol=zeros(1,256);
for x=1:shape(1)
    for y=1:shape(2)
        tol(gray(x,y)+1)=tol(gray(x,y)+1)+1;
    end
end
%����Ҷ�ֱ��ͼ
bar(tol);
%���ÿ�����صĸ���
tol=tol/shape(1)/shape(2);
%����ʷֲ�
for x=2:256
    tol(x)=tol(x)+tol(x-1);
end
%bar(tol);
%ֱ��ͼ���⻯
%������ͼ�ĻҶ�ֵ
tol=int16(255*tol+0.5);
%��ȥ���ظ�Ԫ�غ�۲�
%tol=unique(tol);
%bar(tol);
%�õ��µĻҶ�ͼ
mgi=zeros(shape(1),shape(2));
%�Ҷ�ӳ��
for x=1:shape(1)
    for y=1:shape(2)
        mgi(x,y)=tol(gray(x,y)+1);
    end
end
%��ʾͼ������õ���ֱ��ͼ���⻯�Ľ��
imshow(mgi,[]);
%ȡ����
gray=rgb2gray(image);
gray=log(im2double(gray)+1e-10);
%��ά��ɢ����Ҷ�仯��Ƶ�����
%����������ɢ����Ҷ�任�ڵõ�Ƶ��ͼ��ʱ��
%�Ѿ�����ʵ�֣�Ϊ�˱��ִ�����������಻���ظ�
F=fftshift(fft2(gray));
%ȡ��״
shape=size(F);
%imshow(log(abs(F)),[]);
%���ɵ���
[x,y] = meshgrid(-shape(1)/2:shape(1)/2,-shape(2)/2:shape(2)/2);
%ѡ�񵹸�˹�˲�����һ�����õ�Ƶ���˲���
H = 0.25*(1-exp(-1*(x.^2+y.^2)/200))+0.25;
%�˲������㣬��Ҫ������״
H = imresize(H,[shape(1),shape(2)]);
%���˲�����Ƶ��ͼ��ˣ����Լ��ٸ��Ӷ�
F = F.*H;
%����Ҷ��任��ͬ���������Ѿ�ʵ�֣����ﲻ���ظ�
F=real(ifft2(ifftshift(F)));
%ȡָ������̬ͬ�˲������һ��
F=exp(F);
%���̬ͬ�˲�ͼ��
imshow(F);