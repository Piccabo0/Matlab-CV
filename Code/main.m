%读入一幅彩色图像
rgb=imread('777.jpg'); 
%subplot(241),imshow(rgb,'InitialMagnification','fit'),title('RGB图像');%显示原图像
r=rgb(:,:,1); g=rgb(:,:,2); b=rgb(:,:,3);%取出RGB的R、G、B各分量 

%R层锐化：
r=im2double(r); %双精度化处理
[rX,rY]=gradient(r); %返回矩阵I梯度值的X和Y分量
r1=sqrt(rX.*rX+rY.*rY); %得到梯度算法结果图像
%figure,subplot(252),imshow(r1,[]);title('gradient value ');
h2=fspecial('sobel');%选择索伯尔算法
r2=imfilter(r,h2);%采用索伯尔算法滤波 
%subplot(253),imshow(r2,[]);title('Sobel滤波');% 显示索伯尔算法结果图像 
h3=fspecial('prewitt');%选择蒲瑞维特算法
r3=imfilter(r,h3);%采用蒲瑞维特算法滤波 
%subplot(254),imshow(r3,[]);title('Prewitt滤波');% 显示蒲瑞维特算法结果图像
h4=fspecial('laplacian'); %选择滤波算法为拉普拉 斯算法
r4=imfilter(r,h4);%采用拉普拉斯算法滤波 
%subplot(255),imshow(r4,[]);title('Laplacian滤波'); %在中下区域显示拉普拉斯算法结果图像
%图层相减
%subplot(256),imshow(r,'InitialMagnification','fit'),title('R图像'); 
r5=r-r1;
%subplot(257),imshow(r5,[]);title('R-gradient value');
r6=r-r2;
%subplot(258),imshow(r6,[]);title('R-Sobel滤波');
r7=r-r3;
%subplot(259),imshow(r7,[]);title('R-prewitt滤波');
r8=r-r4;
%subplot(2,5,10),imshow(r8,[]);title('R-Laplacian滤波');
%G层锐化：
g=im2double(g); %双精度化处理
[gX,gY]=gradient(g); %返回矩阵I梯度值的X和Y分量
g1=sqrt(gX.*gX+gY.*gY); %得到梯度算法结果图像
%figure,subplot(252),imshow(g1,[]);title('gradient value ');
h2=fspecial('sobel');%选择索伯尔算法
g2=imfilter(g,h2);%采用索伯尔算法滤波 
%subplot(253),imshow(g2,[]);title('Sobel滤波');% 显示索伯尔算法结果图像 
h3=fspecial('prewitt');%选择蒲瑞维特算法
g3=imfilter(g,h3);%采用蒲瑞维特算法滤波 
%subplot(254),imshow(g3,[]);title('Prewitt滤波');% 显示蒲瑞维特算法结果图像
h4=fspecial('laplacian'); %选择滤波算法为拉普拉 斯算法
g4=imfilter(g,h4);%采用拉普拉斯算法滤波 
%subplot(255),imshow(g4,[]);title('Laplacian滤波'); %在中下区域显示拉普拉斯算法结果图像
%图层相减
%subplot(256),imshow(g,'InitialMagnification','fit'),title('G图像'); 
g5=g-g1;
%subplot(257),imshow(g5,[]);title('G-gradient value');
g6=g-g2;
%subplot(258),imshow(g6,[]);title('G-Sobel滤波');
g7=g-g3;
%subplot(259),imshow(g7,[]);title('G-prewitt滤波');
g8=g-g4;
%subplot(2,5,10),imshow(g8,[]);title('G-Laplacian滤波');

%B层锐化：
b=im2double(b); %双精度化处理
[bX,bY]=gradient(b); %返回矩阵I梯度值的X和Y分量
b1=sqrt(bX.*bX+bY.*bY); %得到梯度算法结果图像
%figure,subplot(252),imshow(b1,[]);title('gradient value ');
h2=fspecial('sobel');%选择索伯尔算法
b2=imfilter(b,h2);%采用索伯尔算法滤波 
%subplot(253),imshow(r2,[]);title('Sobel滤波');% 显示索伯尔算法结果图像 
h3=fspecial('prewitt');%选择蒲瑞维特算法
b3=imfilter(b,h3);%采用蒲瑞维特算法滤波 
%subplot(254),imshow(b3,[]);title('Prewitt滤波');% 显示蒲瑞维特算法结果图像
h4=fspecial('laplacian'); %选择滤波算法为拉普拉 斯算法
b4=imfilter(b,h4);%采用拉普拉斯算法滤波 
%subplot(255),imshow(b4,[]);title('Laplacian滤波'); %在中下区域显示拉普拉斯算法结果图像
%图层相减
%subplot(256),imshow(b,'InitialMagnification','fit'),title('B图像'); 
b5=b-b1;
%subplot(257),imshow(b5,[]);title('B-gradient value');
b6=b-b2;
%subplot(258),imshow(b6,[]);title('B-Sobel滤波');
b7=b-b3;
%subplot(259),imshow(b7,[]);title('B-prewitt滤波');
b8=b-b4;
%subplot(2,5,10),imshow(b8,[]);title('B-Laplacian滤波');

rgb1(:,:,1)=r5;
rgb1(:,:,2)=g5;
rgb1(:,:,3)=b5;
%figure;imshow(rgb1); 
%{
figure;subplot(141);imhist(rgb1);title('RGB-灰度直方图')
subplot(142);imhist(r5);title('R-灰度直方图');
subplot(143);imhist(g5);title('G-灰度直方图');
subplot(144);imhist(b5);title('B-灰度直方图');
%} 
level= graythresh(r5);%采用最大类间方差法自动求取阈值
I1=imbinarize(r5,level);%利用所得到的阈值分割图像
figure,imshow(I1);%显示分割后的二值图像
%{
[M,N]=size(r5);
for i=1:M
     for j=1:N
         if(r5(i,j)<0.75)
             rgb1(i,j,1)=255; rgb1(i,j,2)=255; rgb1(i,j,3)=255;
         %else r1(i,j)=r(i,j); g1(i,j)=b(i,j); b1(i,j)=b(i,j);
         end 
      end
end
 r6=rgb1(:,:,1); g6=rgb1(:,:,2); b6=rgb1(:,:,3);
 figure,imshow(rgb1);
%}
%{
figure;subplot(141);imhist(rgb1);title('RGB-灰度直方图')
subplot(142);imhist(r6);title('R-灰度直方图');
subplot(143);imhist(g6);title('G-灰度直方图');
subplot(144);imhist(b6);title('B-灰度直方图');

rgb2=rgb1;
figure;imshow(rgb2);
[m,n,d]=size(rgb1);
for i=1:m
    for j=1:n
        %加—>绿色少
        if(rgb1(i,j,1)>=0.83 && rgb1(i,j,1)<=1)&&(rgb1(i,j,2)>=0.7 && rgb1(i,j,2)<=1)
            rgb2(i,j,1)=rgb1(i,j,1);
            rgb2(i,j,2)=rgb1(i,j,2);
            rgb2(i,j,3)=rgb1(i,j,3);
        else
            rgb2(i,j,1)=255;
            rgb2(i,j,2)=255;
            rgb2(i,j,3)=255;
        end
    end
end
figure,imshow(rgb2);
 %}