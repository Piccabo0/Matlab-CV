%����һ����ɫͼ��
rgb=imread('777.jpg'); 
%subplot(241),imshow(rgb,'InitialMagnification','fit'),title('RGBͼ��');%��ʾԭͼ��
r=rgb(:,:,1); g=rgb(:,:,2); b=rgb(:,:,3);%ȡ��RGB��R��G��B������ 

%R���񻯣�
r=im2double(r); %˫���Ȼ�����
[rX,rY]=gradient(r); %���ؾ���I�ݶ�ֵ��X��Y����
r1=sqrt(rX.*rX+rY.*rY); %�õ��ݶ��㷨���ͼ��
%figure,subplot(252),imshow(r1,[]);title('gradient value ');
h2=fspecial('sobel');%ѡ���������㷨
r2=imfilter(r,h2);%�����������㷨�˲� 
%subplot(253),imshow(r2,[]);title('Sobel�˲�');% ��ʾ�������㷨���ͼ�� 
h3=fspecial('prewitt');%ѡ������ά���㷨
r3=imfilter(r,h3);%��������ά���㷨�˲� 
%subplot(254),imshow(r3,[]);title('Prewitt�˲�');% ��ʾ����ά���㷨���ͼ��
h4=fspecial('laplacian'); %ѡ���˲��㷨Ϊ������ ˹�㷨
r4=imfilter(r,h4);%����������˹�㷨�˲� 
%subplot(255),imshow(r4,[]);title('Laplacian�˲�'); %������������ʾ������˹�㷨���ͼ��
%ͼ�����
%subplot(256),imshow(r,'InitialMagnification','fit'),title('Rͼ��'); 
r5=r-r1;
%subplot(257),imshow(r5,[]);title('R-gradient value');
r6=r-r2;
%subplot(258),imshow(r6,[]);title('R-Sobel�˲�');
r7=r-r3;
%subplot(259),imshow(r7,[]);title('R-prewitt�˲�');
r8=r-r4;
%subplot(2,5,10),imshow(r8,[]);title('R-Laplacian�˲�');
%G���񻯣�
g=im2double(g); %˫���Ȼ�����
[gX,gY]=gradient(g); %���ؾ���I�ݶ�ֵ��X��Y����
g1=sqrt(gX.*gX+gY.*gY); %�õ��ݶ��㷨���ͼ��
%figure,subplot(252),imshow(g1,[]);title('gradient value ');
h2=fspecial('sobel');%ѡ���������㷨
g2=imfilter(g,h2);%�����������㷨�˲� 
%subplot(253),imshow(g2,[]);title('Sobel�˲�');% ��ʾ�������㷨���ͼ�� 
h3=fspecial('prewitt');%ѡ������ά���㷨
g3=imfilter(g,h3);%��������ά���㷨�˲� 
%subplot(254),imshow(g3,[]);title('Prewitt�˲�');% ��ʾ����ά���㷨���ͼ��
h4=fspecial('laplacian'); %ѡ���˲��㷨Ϊ������ ˹�㷨
g4=imfilter(g,h4);%����������˹�㷨�˲� 
%subplot(255),imshow(g4,[]);title('Laplacian�˲�'); %������������ʾ������˹�㷨���ͼ��
%ͼ�����
%subplot(256),imshow(g,'InitialMagnification','fit'),title('Gͼ��'); 
g5=g-g1;
%subplot(257),imshow(g5,[]);title('G-gradient value');
g6=g-g2;
%subplot(258),imshow(g6,[]);title('G-Sobel�˲�');
g7=g-g3;
%subplot(259),imshow(g7,[]);title('G-prewitt�˲�');
g8=g-g4;
%subplot(2,5,10),imshow(g8,[]);title('G-Laplacian�˲�');

%B���񻯣�
b=im2double(b); %˫���Ȼ�����
[bX,bY]=gradient(b); %���ؾ���I�ݶ�ֵ��X��Y����
b1=sqrt(bX.*bX+bY.*bY); %�õ��ݶ��㷨���ͼ��
%figure,subplot(252),imshow(b1,[]);title('gradient value ');
h2=fspecial('sobel');%ѡ���������㷨
b2=imfilter(b,h2);%�����������㷨�˲� 
%subplot(253),imshow(r2,[]);title('Sobel�˲�');% ��ʾ�������㷨���ͼ�� 
h3=fspecial('prewitt');%ѡ������ά���㷨
b3=imfilter(b,h3);%��������ά���㷨�˲� 
%subplot(254),imshow(b3,[]);title('Prewitt�˲�');% ��ʾ����ά���㷨���ͼ��
h4=fspecial('laplacian'); %ѡ���˲��㷨Ϊ������ ˹�㷨
b4=imfilter(b,h4);%����������˹�㷨�˲� 
%subplot(255),imshow(b4,[]);title('Laplacian�˲�'); %������������ʾ������˹�㷨���ͼ��
%ͼ�����
%subplot(256),imshow(b,'InitialMagnification','fit'),title('Bͼ��'); 
b5=b-b1;
%subplot(257),imshow(b5,[]);title('B-gradient value');
b6=b-b2;
%subplot(258),imshow(b6,[]);title('B-Sobel�˲�');
b7=b-b3;
%subplot(259),imshow(b7,[]);title('B-prewitt�˲�');
b8=b-b4;
%subplot(2,5,10),imshow(b8,[]);title('B-Laplacian�˲�');

rgb1(:,:,1)=r5;
rgb1(:,:,2)=g5;
rgb1(:,:,3)=b5;
%figure;imshow(rgb1); 
%{
figure;subplot(141);imhist(rgb1);title('RGB-�Ҷ�ֱ��ͼ')
subplot(142);imhist(r5);title('R-�Ҷ�ֱ��ͼ');
subplot(143);imhist(g5);title('G-�Ҷ�ֱ��ͼ');
subplot(144);imhist(b5);title('B-�Ҷ�ֱ��ͼ');
%} 
level= graythresh(r5);%���������䷽��Զ���ȡ��ֵ
I1=imbinarize(r5,level);%�������õ�����ֵ�ָ�ͼ��
figure,imshow(I1);%��ʾ�ָ��Ķ�ֵͼ��
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
figure;subplot(141);imhist(rgb1);title('RGB-�Ҷ�ֱ��ͼ')
subplot(142);imhist(r6);title('R-�Ҷ�ֱ��ͼ');
subplot(143);imhist(g6);title('G-�Ҷ�ֱ��ͼ');
subplot(144);imhist(b6);title('B-�Ҷ�ֱ��ͼ');

rgb2=rgb1;
figure;imshow(rgb2);
[m,n,d]=size(rgb1);
for i=1:m
    for j=1:n
        %�ӡ�>��ɫ��
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