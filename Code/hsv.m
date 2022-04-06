rgb = imread('777.jpg');

%r=a(:,:,1);g=a(:,:,2);b=a(:,:,3);
%subplot(131);imhist(r);title('R-�Ҷ�ֱ��ͼ');
%subplot(132);imhist(g);title('G-�Ҷ�ֱ��ͼ');
%subplot(133);imhist(b);title('B-�Ҷ�ֱ��ͼ');

% hsi = rgb2hsi(rgb)��һ��RGBͼ��ת��ΪHSIͼ��
% ����ͼ����һ����ɫ���ص�M��N��3�����飬
% ����ÿһ����ɫ���ض����ض��ռ�λ�õĲ�ɫͼ���ж�Ӧ�졢�̡�������������
% �������е�RGB�����Ǿ���ģ���ôHSIת������δ����ġ�
% ����ͼ�������double��ȡֵ��Χ��[0, 1]����uint8�� uint16��
%
% ���HSIͼ����double��
% ����hsi(:, :, 1)��ɫ�ȷ��������ķ�Χ�ǳ���2*pi���[0, 1]��
% hsi(:, :, 2)�Ǳ��Ͷȷ�������Χ��[0, 1]��
% hsi(:, :, 3)�����ȷ�������Χ��[0, 1]��
 
% ��ȡͼ�����
rgb = im2double(rgb);
r = rgb(:, :, 1);
g = rgb(:, :, 2);
b = rgb(:, :, 3);
 
% ִ��ת������
num = 0.5*((r - g) + (r - b));
den = sqrt((r - g).^2 + (r - b).*(g - b));
theta = acos(num./(den + eps)); %��ֹ����Ϊ0
 
H = theta;
H(b > g) = 2*pi - H(b > g);
H = H/(2*pi);
 
num = min(min(r, g), b);
den = r + g + b;
den(den == 0) = eps; %��ֹ����Ϊ0
S = 1 - 3.* num./den;
 
H(S == 0) = 0;
 
I = (r + g + b)/3;
 
% ��3���������ϳ�Ϊһ��HSIͼ��
hsi = cat(3, H, S, I);
hsi1 = H;   
hsi2 = S;
hsi3 = I;
figure,imshow(hsi);
figure,subplot(131),imshow(hsi1);title('H����');
subplot(132),imshow(hsi2);title('S����');
subplot(133),imshow(hsi3);title('I����');
figure,subplot(131),imhist(hsi1);title('H�Ҷ�ֱ��ͼ');
subplot(132),imhist(hsi2);title('S�Ҷ�ֱ��ͼ');
subplot(133),imhist(hsi3);title('I�Ҷ�ֱ��ͼ');
rgb1(:,:,1)=hsi1;
rgb1(:,:,2)=hsi2;
rgb1(:,:,3)=hsi3;
[M,N]=size(hsi1);
for i=1:M
    for j=1:N
        if(hsi1(i,j)>0.20392)
            rgb1(i,j,1)=255; rgb1(i,j,2)=255; rgb1(i,j,3)=255;
        end
    end
end
figure;imshow(rgb1);



%figure,imshow(hsi);


%{
[M,N,O] = size(Image);
[h,s,v] = rgb2hsv(Image);

H = h; S = s; V = v;
h = h*360;  
%��hsv�ռ�ǵȼ��������
%  h������16����
%  s������4����
%  v������4����
for i = 1:M
    for j = 1:N
        if h(i,j)<=15||h(i,j)>345
            H(i,j) = 0;
        end
        if h(i,j)<=25&&h(i,j)>15
            H(i,j) = 1;
        end
        if h(i,j)<=45&&h(i,j)>25
            H(i,j) = 2;
        end
        if h(i,j)<=55&&h(i,j)>45
            H(i,j) = 3;
        end
        if h(i,j)<=80&&h(i,j)>55
            H(i,j) = 4;
        end
        if h(i,j)<=108&&h(i,j)>80
            H(i,j) = 5;
        end
        if h(i,j)<=140&&h(i,j)>108
            H(i,j) = 6;
        end
        if h(i,j)<=165&&h(i,j)>140
            H(i,j) = 7;
        end
        if h(i,j)<=190&&h(i,j)>165
            H(i,j) = 8;
        end
        if h(i,j)<=220&&h(i,j)>190
            H(i,j) = 9;
        end
        if h(i,j)<=255&&h(i,j)>220
            H(i,j) = 10;
        end
        if h(i,j)<=275&&h(i,j)>255
            H(i,j) = 11;
        end
        if h(i,j)<=290&&h(i,j)>275
            H(i,j) = 12;
        end
        if h(i,j)<=316&&h(i,j)>290
            H(i,j) = 13;
        end
        if h(i,j)<=330&&h(i,j)>316
            H(i,j) = 14;
        end
        if h(i,j)<=345&&h(i,j)>330
            H(i,j) = 15;
        end
    end
end
for i = 1:M
    for j = 1:N
        if s(i,j)<=0.15&&s(i,j)>0
            S(i,j) = 1;
        end
        if s(i,j)<=0.4&&s(i,j)>0.15
            S(i,j) = 2;
        end
        if s(i,j)<=0.75&&s(i,j)>0.4
            S(i,j) = 3;
        end
        if s(i,j)<=1&&s(i,j)>0.75
            S(i,j) = 4;
        end
    end
end
for i = 1:M
    for j = 1:N
        if v(i,j)<=0.15&&v(i,j)>0
            V(i,j) = 1;
        end
        if v(i,j)<=0.4&&v(i,j)>0.15
            V(i,j) = 2;
        end
        if v(i,j)<=0.75&&v(i,j)>0.4
            V(i,j) = 3;
        end
        if v(i,j)<=1&&v(i,j)>0.75
            V(i,j) = 4;
        end
    end
end


% ����4*16��ά������H-S����
Hist = zeros(16,4);
for i = 1:M
    for j = 1:N
        for k = 1:16
            for l = 1:4
                if  l==S(i,j)&& k==H(i,j)+1
                    Hist(k,l) = Hist(k,l)+1;
                end
            end
        end
    end
end
for k = 1:16
    for l =1:4
        His((k-1)*4+l) = Hist(k,l);%ת��Ϊһά����
    end
end
His = His/sum(His)*1000;

% �ֹ����Ʋ�ɫͼ��ֱ��ͼ
% hist_h
m=0;
for j = 1:300
    if rem(j,16)==1 && m<16
        for k = 0:15
            for i = 1:200
                hist_h(i,j+k) = m;
            end            
        end
        m = m+1;
    end
end
% hist_s
m=0;
for j = 1:300
    if rem(j,4) == 1 && m<64
        n = rem(m,4);
        for k = 0:3              
            for i =1:200              
                hist_s(i,j+k) = n+1;                
            end                     
        end
        m = m+1; 
    end    
end
% hist_v
for j = 1:256
    for i = 1:200
        hist_v(i,j) = 0.98;
    end
end
% ��His��ֵ��hist_v
for k = 1:64
    for j = 1:256
        if floor((j-1)/4) == k
            for i = 1:200
                if i<200-His(k+1)%i>His(k+1)%
                    hist_v(i,j) = 0;
                end
            end
        end
    end
end

%��h��s��v����ͼ�ϲ�ת��ΪRGBģʽ

I_H = hsv2rgb(hist_h/16,hist_s/4,hist_v);

Image1=im2double(Image);
[M,N,d]=size(V);
for i=1:M
    for j=i:N
        if(V(i,j)==1)
            Image1(i,j)=V(i,j);
        else
            Image1(i,j)=0;
        end
    end
end


% ��ͼ��ʾ 
figure;
subplot(3,3,1),imshow(Image),title('ԭͼ');
subplot(3,3,2),imshow(H,[]),title('H����ͼ');
subplot(3,3,3),imshow(S,[]),title('S����ͼ');
subplot(3,3,4),imshow(V,[]),title('V����ͼ');
%subplot(3,3,5),imshow(I_H,[]),title('H-Sֱ��ͼ');
subplot(3,3,6),imshow(Image1,[]),title('V�ָ�');
%figure,imshow(I_H);
%}