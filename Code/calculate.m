I=imread('777.png');
I=rgb2gray(I);
%figure,subplot(131),imshow(I),title('ԭʼͼ��');%��ʾ�ûҶ�ͼ��
B=imbinarize(I,graythresh(I));%�ԸûҶ�ͼ������Զ���ֵ
%B1=~B;%��ֵͼ��ȡ��
%subplot(132),imshow(B1),title('��ֵͼ��');%��ʾȡ�����ֵͼ��
B=imerode(B,strel('disk',5));%��ʴ����
%figure,imshow(B1),title('���ͼ��');
[x,n]=bwlabel(B);%������
stats=regionprops(x, 'all');% �����������ĵ�19����������
X=label2rgb(x); %�Ա��ͼ����α��ɫ�任
%imshow(X),title('���ͼ��'); %��ʾα�ʱ任��ı��ͼ��
%imshow(B1),title('���ͼ��');
B1=zeros(size(B));
for i=1:n %������㲢��ʾ����������������������ֵ
    ar=stats(i).Area;
    disp(ar);
    if ar>1000
        [p,q]=find(x==i);
        B2=bwselect(B,q,p,8);%��ȡֻ�и�����Ķ�ֵͼ��
        B1=(B1)|(B2);
        %��ȡ������������Ҫ�������Ķ�ֵ
    end
end
figure,imshow(B1);
%{
I=rgb2gray(I);
%��������ͼ���Ƶ��
s=fftshift(fft2(I));%��ά����Ҷ�任
s=abs(s);%����Ƶ��s�ľ���ֵ
[nc,nr]=size(s);%����Ƶ��s�ĳߴ�
x0=nc/2+1;y0=nr/2+1;%����ԭ������
rmax=min(nc,nr)/2-1;%Բ�İ뾶�����ȡֵ
thetha=91:270;%��Բ
srad=zeros(1,rmax);%����1*rmax����ֵ��Ϊ0������
srad(1)=s(x0,y0);%ԭ���Ƶ��ֵ
for r=1:rmax %��1��rmax��Բ�İ뾶
    [x,y]=pol2cart(thetha,r);%�Ӽ�����ת����ֱ������
    x=round(x)+x0;
    y=round(y)+y0;
    for j=1:length(x) %��x�ĳߴ緶Χ��
        srad(r)=sum(s(sub2ind(size(s),x,y)));%��Ƶ��s������ߴ�
        size(s)%�����ϵĸ�������(x,y)������������������ͣ���ʽ
    end
end
[x,y]=pol2cart(thetha,rmax);%�Ӽ�����ת����ֱ������
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
            xr(i)=D(1);%ʹxr��ֵ������size(s)����
        end
    end
    for i=1:length(yr)
        if(yr(i)>D(2))
            yr(i)=D(2);%ʹyr��ֵ������size(s)��
        end
    end
    for j=1:length(xr)
        sang(th)=sum(s(sub2ind(size(s),xr,yr)));%ʽ
    end
end
for j=1:length(xr)
    sang(th)=sum(s(sub2ind(size(s),xr,yr)));%ʽ
end
s=mat2gray(log(s+1));
figure,imshow(s);%Ƶ��ͼ
figure,plot(srad);%Ƶ�׺���S(r)����ͼ
figure,plot(sang);%Ƶ�׺���S(��)����ͼ
%}

%{
I=imresize(I,1/40); %��ͼ��ߴ��Ա���1/40��С
[M,N]=size(I);%��ȡ��С���ͼ��ߴ�
%Ϊ�˼��ټ���������ԭʼͼ��Ҷȼ���256������16��
for i = 1:M
    for j = 1:N
        for n = 1:16
            if (n-1)*16<=I(i,j) && I(i,j)<=(n-1)*16+15
                I(i,j) = n-1;
            end
        end
    end
end
%�����ĸ���������P(1 ,0)�� P(-1,1) ��P(0,1)��P(1,1)
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
% �ԻҶȹ��������һ��
for n = 1:4
    P(:,:,n) = P(:,:,n)/sum(sum(P(:,:,n)));
end
H = zeros(1,4);%�س�ʼ��
CON = H;%���Ծس�ʼ��
Ux = H; Uy = H; %������Ц�x�ͦ�y��ʼ��
deltaX= H; deltaY = H; %������Ц�x�ͦ�y��ʼ��
COR =H; %�����ʼ��
L=H;
for n = 1:4
    ASM(n) = sum(sum(P(:,:,n).^2)); %%����
    for i = 1:16
        for j = 1:16
            if P(i,j,n)~=0
                H( n)= -P(i,j,n )*log(P(i,j,n ))+H(n ) ;%��
            end
            CON(n) = (i-j)^2*P(i,j,n)+CON(n); %���Ծ�
            Ux(n) = i*P(i,j,n)+Ux(n); %������Ц�x
            Uy(n) = j*P(i,j,n)+Uy(n); %������Ц�y
        end
    end
end
for n = 1:4
    for i = 1:16
        for j = 1:16
            deltaX(n) = (i-Ux(n))^2*P(i,j,n)+deltaX(n); %������Ц�x
            deltaY(n) = (j-Uy(n))^2*P(i,j,n)+deltaY(n); %������Ц�y
            COR(n) = i*j*P(i,j,n)+COR(n); %�����
            L(n)=P(i,j,n)^2/(1+(i-j)^2)+L(n); %����
        end
    end
    COR(n) = (COR(n)-Ux(n)*Uy(n))/deltaX(n)/deltaY(n);
end
sprintf('0,45,90,135�����ϵ���������Ϊ�� %f, %f, %f, %f',ASM(1),ASM(2),ASM(3),ASM(4)) %�������
sprintf('0,45,90,135�����ϵ�������Ϊ�� %f, %f, %f, %f',H(1),H(2),H(3),H(4)) %�����
sprintf('0,45,90,135�����ϵĹ��Ծ�����Ϊ�� %f, %f, %f, %f', CON(1), CON(2), CON(3), CON(4)) %������Ծ�
sprintf('0,45,90,135�����ϵ����������Ϊ�� %f, %f, %f, %f', COR(1), COR(2), COR(3), COR(4)) %��������
sprintf('0,45,90,135�����ϵ���������Ϊ�� %f, %f, %f, %f',L(1),L(2),L(3),L(4)) %�������
%���������ء����Ծء�����ԡ�����ľ�ֵ�ͱ�׼����Ϊ���յ���������
a1 = mean(ASM); b1 = sqrt(cov(ASM));
a2 = mean(H); b2 = sqrt(cov(H));
a3 = mean(CON); b3 = sqrt(cov(CON));
a4 = mean(COR);b4 = sqrt(cov(COR));
a5 = mean(L); b5 = sqrt(cov(L));
sprintf('�����ľ�ֵ�ͱ�׼��ֱ�Ϊ�� %f, %f',a1,b1) % ��������ľ�ֵ�ͱ�׼��
sprintf('�صľ�ֵ�ͱ�׼��ֱ�Ϊ�� %f, %f',a2,b2) % ����صľ�ֵ�ͱ�׼��
sprintf('���Ծصľ�ֵ�ͱ�׼��ֱ�Ϊ�� %f, %f',a3,b3) % ������Ծصľ�ֵ�ͱ�׼��
sprintf('����Եľ�ֵ�ͱ�׼��ֱ�Ϊ�� %f, %f',a4,b4) % �������Եľ�ֵ�ͱ�׼��
sprintf('����ľ�ֵ�ͱ�׼��ֱ�Ϊ�� %f, %f',a5,b5) % �������ľ�ֵ�ͱ�׼��
%}

%{
h=imhist(I);%������ͼ���ֱ��ͼ����h������
figure,plot(h); %��ʾ����ͼ���ֱ��ͼ
h=h/sum(h);%��һ��
L=length(h);%����Ҷȼ�
L=L-1;
h=h(:);%ת��Ϊ������
rad=0:L;%���������
rad=rad./L;%��һ��
m=rad*h;%��ֵ
rad=rad-m;
stm=zeros(1,3);
stm(1)=m;
for j=2:3
stm(j)=(rad.^j)*h;%����n�׾�
end
usm(1)=stm(1)*L;%һ�׾�
usm(2)=stm(2)*L^2;%���׾�
usm(3)=stm(3)*L^3;%���׾�
st(1)=usm(1); %��ֵ
st(2)=usm(2).^0.5 ;%��׼��
st(3)=1-1/(1+usm(2)); %ƽ����
st(4)=usm(3)/(L^2); %��һ�������׾�
st(5)=sum(h.^2) ;%һ����
st(6)=-sum(h.*log2(h+eps)); %��
for i=1:6
    disp(st(i));
end
%}

%ֱ��ͼ
%{
R=I(:,:,1);%��ȡ��ɫ����
G=I(:,:,2);%��ȡ��ɫ����
B=I(:,:,3);%��ȡ��ɫ����
figure;
subplot(1,3,1),imhist(R);title('R�����Ҷ�ֱ��ͼ');
subplot(1,3,2),imhist(G);title('G�����Ҷ�ֱ��ͼ');
subplot(1,3,3),imhist(B);title('B�����Ҷ�ֱ��ͼ');

siz=size(I);
I1=reshape(I,siz(1)*siz(2),siz(3));  % ÿ����ɫͨ����Ϊһ��
I1=double(I1);
[N,X]=hist(I1,[0:5:255]);   
% �����ҪС���ο�һ�㣬���������ٵ㣬���԰Ѳ����Ĵ󣬱���0:5:255
figure,bar(X,N(:,[1 2 3]));
legend('R�����Ҷȷֲ�','G�����Ҷȷֲ�','B�����Ҷȷֲ�');
% ����ͼ����N(:,[3 2 1])����ΪĬ�ϻ�ͼ��ʱ����õ���ɫ˳��Ϊb,g,r,c,m,y,k��
%��ͼƬ��rgb˳�������෴�����԰�ͼƬ�е�˳�򵹹�����
%��ͼƬ��ɫͨ��������ʱ����ɫһ��
xlim([0 255]);
%}

%{
hold on
%plot(X,N(:,[3 2 1]));    % �ϱ߽�����
hold off
%}


%{
I=imread('7777.png');%����һ���Ҷ�ͼ��
I=rgb2gray(I);
subplot(131),imshow(I),title('ԭʼͼ��');%��ʾ�ûҶ�ͼ��
B=imbinarize(I,graythresh(I));%�ԸûҶ�ͼ������Զ���ֵ
B1=~B;%��ֵͼ��ȡ��
subplot(132),imshow(B1),title('��ֵͼ��');%��ʾȡ�����ֵͼ��
[x,n]=bwlabel(B1);%������
stats=regionprops(x, 'all');% �����������ĵ�19����������
X=label2rgb(x); %�Ա��ͼ����α��ɫ�任
subplot(133),imshow(X),title('���ͼ��'); %��ʾα�ʱ任��ı��ͼ��
A=stats.Area ;%ȡ�������ֵ
C=stats.Centroid; %ȡ��������ֵ
EN=stats.EulerNumber; %ȡ��ŷ����
Max=stats.MajorAxisLength ;%ȡ�����᳤��
Xite=stats.Orientation;%ȡ�����᷽���
E=stats.Eccentricity ;%ȡ��������
%}
%{
for i=1:n %������㲢��ʾ����������������������ֵ
disp(stats(i).Area);disp(stats(i).Centroid);
end

disp(A);
disp(C);
disp(EN);
disp(Max);
disp(Xite);
disp(E);
%}