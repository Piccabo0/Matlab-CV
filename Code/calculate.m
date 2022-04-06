I=imread('777.png');
I=rgb2gray(I);
%figure,subplot(131),imshow(I),title('原始图像');%显示该灰度图像
B=imbinarize(I,graythresh(I));%对该灰度图像进行自动阈值
%B1=~B;%二值图像取反
%subplot(132),imshow(B1),title('二值图像');%显示取反后二值图像
B=imerode(B,strel('disk',5));%腐蚀运算
%figure,imshow(B1),title('标记图像');
[x,n]=bwlabel(B);%区域标记
stats=regionprops(x, 'all');% 求出面积、重心等19个特征参数
X=label2rgb(x); %对标记图像做伪彩色变换
%imshow(X),title('标记图像'); %显示伪彩变换后的标记图像
%imshow(B1),title('标记图像');
B1=zeros(size(B));
for i=1:n %逐个计算并显示各个区域的面积和重心坐标值
    ar=stats(i).Area;
    disp(ar);
    if ar>1000
        [p,q]=find(x==i);
        B2=bwselect(B,q,p,8);%获取只有该区域的二值图像
        B1=(B1)|(B2);
        %获取所有满足上述要求的区域的二值
    end
end
figure,imshow(B1);
%{
I=rgb2gray(I);
%计算输入图像的频谱
s=fftshift(fft2(I));%二维傅里叶变换
s=abs(s);%计算频谱s的绝对值
[nc,nr]=size(s);%计算频谱s的尺寸
x0=nc/2+1;y0=nr/2+1;%计算原点坐标
rmax=min(nc,nr)/2-1;%圆的半径的最大取值
thetha=91:270;%半圆
srad=zeros(1,rmax);%创建1*rmax的数值皆为0的数组
srad(1)=s(x0,y0);%原点的频谱值
for r=1:rmax %从1到rmax的圆的半径
    [x,y]=pol2cart(thetha,r);%从极坐标转换到直角坐标
    x=round(x)+x0;
    y=round(y)+y0;
    for j=1:length(x) %在x的尺寸范围内
        srad(r)=sum(s(sub2ind(size(s),x,y)));%对频谱s按照其尺寸
        size(s)%给其上的各个坐标(x,y)做线性索引并对其求和，即式
    end
end
[x,y]=pol2cart(thetha,rmax);%从极坐标转换到直角坐标
x=round(x)+x0; y=round(y)+y0;
sang=zeros(1,length(x));
for th=1:length(x)
    vx=abs(x(th)-x0);
    vy=abs(y(th)-y0);
    if((vx==0 & vy==0))
        xr=x0;yr=y0;
    else
        m=vy/vx;
        xr=x0:x(th);
        yr=round(y(th)+m*(xr-x0));
    end
    D = size(s);
    for i=1:length(xr)
        if(xr(i)>D(1))
            xr(i)=D(1);%使xr的值限制在size(s)以内
        end
    end
    for i=1:length(yr)
        if(yr(i)>D(2))
            yr(i)=D(2);%使yr的值限制在size(s)内
        end
    end
    for j=1:length(xr)
        sang(th)=sum(s(sub2ind(size(s),xr,yr)));%式
    end
end
for j=1:length(xr)
    sang(th)=sum(s(sub2ind(size(s),xr,yr)));%式
end
s=mat2gray(log(s+1));
figure,imshow(s);%频谱图
figure,plot(srad);%频谱函数S(r)曲线图
figure,plot(sang);%频谱函数S(θ)曲线图
%}

%{
I=imresize(I,1/40); %将图像尺寸以比例1/40缩小
[M,N]=size(I);%获取缩小后的图像尺寸
%为了减少计算量，对原始图像灰度级从256级降成16级
for i = 1:M
    for j = 1:N
        for n = 1:16
            if (n-1)*16<=I(i,j) && I(i,j)<=(n-1)*16+15
                I(i,j) = n-1;
            end
        end
    end
end
%计算四个共生矩阵P(1 ,0)、 P(-1,1) 、P(0,1)、P(1,1)
P = zeros(16,16,4);
for m = 1:16
    for n = 1:16
        for i = 1:M
            for j = 1:N
                if j<N & I(i,j)==m-1 & I(i,j+1)==n-1
                    P(m,n,1) = P(m,n,1)+1;
                end
                if i>1&j<N & I(i,j)==m-1 & I(i-1,j+1)==n-1
                    P(m,n,2) = P(m,n,2)+1;
                end
                if i<M & I(i,j)==m-1 & I(i+1,j)==n-1
                    P(m,n,3) = P(m,n,3)+1;
                end
                if i<M & j<N & I(i,j)==m-1 & I(i+1,j+1)==n-1
                    P(m,n,4) = P(m,n,4)+1;
                end
            end
        end
    end
end
% 对灰度共生矩阵归一化
for n = 1:4
    P(:,:,n) = P(:,:,n)/sum(sum(P(:,:,n)));
end
H = zeros(1,4);%熵初始化
CON = H;%惯性矩初始化
Ux = H; Uy = H; %相关性中μx和μy初始化
deltaX= H; deltaY = H; %相关性中σx和σy初始化
COR =H; %逆差距初始化
L=H;
for n = 1:4
    ASM(n) = sum(sum(P(:,:,n).^2)); %%能量
    for i = 1:16
        for j = 1:16
            if P(i,j,n)~=0
                H( n)= -P(i,j,n )*log(P(i,j,n ))+H(n ) ;%熵
            end
            CON(n) = (i-j)^2*P(i,j,n)+CON(n); %惯性矩
            Ux(n) = i*P(i,j,n)+Ux(n); %相关性中μx
            Uy(n) = j*P(i,j,n)+Uy(n); %相关性中μy
        end
    end
end
for n = 1:4
    for i = 1:16
        for j = 1:16
            deltaX(n) = (i-Ux(n))^2*P(i,j,n)+deltaX(n); %相关性中σx
            deltaY(n) = (j-Uy(n))^2*P(i,j,n)+deltaY(n); %相关性中σy
            COR(n) = i*j*P(i,j,n)+COR(n); %相关性
            L(n)=P(i,j,n)^2/(1+(i-j)^2)+L(n); %逆差距
        end
    end
    COR(n) = (COR(n)-Ux(n)*Uy(n))/deltaX(n)/deltaY(n);
end
sprintf('0,45,90,135方向上的能量依次为： %f, %f, %f, %f',ASM(1),ASM(2),ASM(3),ASM(4)) %输出能量
sprintf('0,45,90,135方向上的熵依次为： %f, %f, %f, %f',H(1),H(2),H(3),H(4)) %输出熵
sprintf('0,45,90,135方向上的惯性矩依次为： %f, %f, %f, %f', CON(1), CON(2), CON(3), CON(4)) %输出惯性矩
sprintf('0,45,90,135方向上的相关性依次为： %f, %f, %f, %f', COR(1), COR(2), COR(3), COR(4)) %输出相关性
sprintf('0,45,90,135方向上的逆差距依次为： %f, %f, %f, %f',L(1),L(2),L(3),L(4)) %输出逆差距
%求能量、熵、惯性矩、相关性、逆差距的均值和标准差作为最终的纹理特征
a1 = mean(ASM); b1 = sqrt(cov(ASM));
a2 = mean(H); b2 = sqrt(cov(H));
a3 = mean(CON); b3 = sqrt(cov(CON));
a4 = mean(COR);b4 = sqrt(cov(COR));
a5 = mean(L); b5 = sqrt(cov(L));
sprintf('能量的均值和标准差分别为： %f, %f',a1,b1) % 输出能量的均值和标准差
sprintf('熵的均值和标准差分别为： %f, %f',a2,b2) % 输出熵的均值和标准差
sprintf('惯性矩的均值和标准差分别为： %f, %f',a3,b3) % 输出惯性矩的均值和标准差
sprintf('相关性的均值和标准差分别为： %f, %f',a4,b4) % 输出相关性的均值和标准差
sprintf('逆差距的均值和标准差分别为： %f, %f',a5,b5) % 输出逆差距的均值和标准差
%}

%{
h=imhist(I);%把纹理图像的直方图代入h变量中
figure,plot(h); %显示纹理图像的直方图
h=h/sum(h);%归一化
L=length(h);%计算灰度级
L=L-1;
h=h(:);%转化为列向量
rad=0:L;%生成随机数
rad=rad./L;%归一化
m=rad*h;%均值
rad=rad-m;
stm=zeros(1,3);
stm(1)=m;
for j=2:3
stm(j)=(rad.^j)*h;%计算n阶矩
end
usm(1)=stm(1)*L;%一阶矩
usm(2)=stm(2)*L^2;%二阶矩
usm(3)=stm(3)*L^3;%三阶矩
st(1)=usm(1); %均值
st(2)=usm(2).^0.5 ;%标准差
st(3)=1-1/(1+usm(2)); %平滑度
st(4)=usm(3)/(L^2); %归一化的三阶矩
st(5)=sum(h.^2) ;%一致性
st(6)=-sum(h.*log2(h+eps)); %熵
for i=1:6
    disp(st(i));
end
%}

%直方图
%{
R=I(:,:,1);%提取红色分量
G=I(:,:,2);%提取绿色分量
B=I(:,:,3);%提取蓝色分量
figure;
subplot(1,3,1),imhist(R);title('R分量灰度直方图');
subplot(1,3,2),imhist(G);title('G分量灰度直方图');
subplot(1,3,3),imhist(B);title('B分量灰度直方图');

siz=size(I);
I1=reshape(I,siz(1)*siz(2),siz(3));  % 每个颜色通道变为一列
I1=double(I1);
[N,X]=hist(I1,[0:5:255]);   
% 如果需要小矩形宽一点，划分区域少点，可以把步长改大，比如0:5:255
figure,bar(X,N(:,[1 2 3]));
legend('R分量灰度分布','G分量灰度分布','B分量灰度分布');
% 柱形图，用N(:,[3 2 1])是因为默认绘图的时候采用的颜色顺序为b,g,r,c,m,y,k，
%跟图片的rgb顺序正好相反，所以把图片列的顺序倒过来，
%让图片颜色通道跟绘制时的颜色一致
xlim([0 255]);
%}

%{
hold on
%plot(X,N(:,[3 2 1]));    % 上边界轮廓
hold off
%}


%{
I=imread('7777.png');%读入一幅灰度图像
I=rgb2gray(I);
subplot(131),imshow(I),title('原始图像');%显示该灰度图像
B=imbinarize(I,graythresh(I));%对该灰度图像进行自动阈值
B1=~B;%二值图像取反
subplot(132),imshow(B1),title('二值图像');%显示取反后二值图像
[x,n]=bwlabel(B1);%区域标记
stats=regionprops(x, 'all');% 求出面积、重心等19个特征参数
X=label2rgb(x); %对标记图像做伪彩色变换
subplot(133),imshow(X),title('标记图像'); %显示伪彩变换后的标记图像
A=stats.Area ;%取出面积数值
C=stats.Centroid; %取出重心数值
EN=stats.EulerNumber; %取出欧拉数
Max=stats.MajorAxisLength ;%取出主轴长度
Xite=stats.Orientation;%取出主轴方向角
E=stats.Eccentricity ;%取出离心率
%}
%{
for i=1:n %逐个计算并显示各个区域的面积和重心坐标值
disp(stats(i).Area);disp(stats(i).Centroid);
end

disp(A);
disp(C);
disp(EN);
disp(Max);
disp(Xite);
disp(E);
%}