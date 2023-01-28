clc; clear; close all; warning off all;

%source nama folder data latih
nama_folder = 'Test';
%membaca file yang berekstensi .jpg
nama_file = dir(fullfile(nama_folder, '*.jpg'));
%membaca jumlah file
jumlah_file = numel(nama_file);

%menginisialisasi variabel

ciri_uji = zeros(jumlah_file,9);

%pengolahan citra terhadap seluruh citra
for n = 1:jumlah_file
     %membaca file citra rgb
    I = imread(fullfile(nama_folder,nama_file(n).name));
   % figure, imshow(I)
    %konversi citra rgb ke citra grayscale
    Imggray = rgb2gray(I);
    %konversi citra grayscale ke citra biner
    bw = imbinarize(Imggray);
    %operasi komplemen - untuk membalikan nilai piksel
    bw = imcomplement(bw);
    %melakukan operasi morfologi filling holes
    bw = imfill(bw,'holes');
    %figure, imshow(bw)
    %melakukan thresholding terhadap komponen red
    K = imbinarize(Imggray,.9);
    %figure, imshow(K)
    %melakukan operasi komplemen
    L = imcomplement(K);
    %figure, imshow(L)
    
    %melakukan operasi morfologi
    %1. closing
    str = strel('disk',5);
    M = imclose(L,str);
    %figure, imshow(M)
    
    %2. filling holes
    N = imfill(M,'holes');
    %figure, imshow(N)
    
    %3. area opening
    O = bwareaopen(N,100);
    %figure, imshow(O)
    
    %ekstraski ciri
    stats = regionprops(O,'Area','Perimeter','Eccentricity');
    area = stats.Area;
    perimeter = stats.Perimeter;
    metric = 4*pi*area/(perimeter^2);
    eccentricity = stats.Eccentricity;
    %ekstraksi ciri warna
    %ekstraksi komponen rgb
    R = I(:,:,1);
    G = I(:,:,2);
    B = I(:,:,3);
    %mengubah nilai piksell background menjadi nol
    R(~bw) = 0;
    G(~bw) = 0;
    B(~bw) = 0;
    RGB = cat(3,R,G,B);
    %menghitung rata2 nlai rgb
    Red = sum(sum(R))/sum(sum(bw));
    Green = sum(sum(R))/sum(sum(bw));
    Blue = sum(sum(R))/sum(sum(bw));
    %ekstraksi Nilai rata2 HSV
    %konversi citra ke HSV
    HSV = rgb2hsv(I);
    %figure, imshow(HSV)
    %ekstraksi komponen HSV
    H = HSV(:,:,1);
    S = HSV(:,:,2);
    V = HSV(:,:,3);
    %mengubah nilai piksel
    H(~bw) = 0;
    S(~bw) = 0;
    V(~bw) = 0;
    %menghitung rata2 nilai hsv
    Hue = sum(sum(H))/sum(sum(bw));
    Saturation = sum(sum(S))/sum(sum(bw));
    Value = sum(sum(V))/sum(sum(bw));
    Luas = sum(sum(bw));
    
    ciri_uji(n,1) = metric;
    ciri_uji(n,2) = eccentricity;
    ciri_uji(n,3) = Red;
    ciri_uji(n,4) = Green;
    ciri_uji(n,5) = Blue;
    ciri_uji(n,6) = Hue;
    ciri_uji(n,7) = Saturation;
    ciri_uji(n,8) = Value;
    ciri_uji(n,9) = Luas;
end    

%menyusun variabel input
kelas_uji = cell(jumlah_file,1);
%menyusun variabel target

for k = 1:4
    kelas_uji{k} = 'Pineaple';
end
for k = 5:8
    kelas_uji{k} = 'PineapleMini';
end
for k = 9:12
    kelas_uji{k} = 'Raspberry';
end
for k = 13:16
    kelas_uji{k} = 'Redcurrant';
end
for k = 17:20
    kelas_uji{k} = 'Strawberry';
end
for k = 21:24
    kelas_uji{k} = 'StrawberryWedge';
end
%membangun arsitektur jaringan syaraf tiruan
load Knn
%membaca nilai keluaran jaringan
hasil_uji = predict(Knn,ciri_uji);


%membaca akurasi
jumlah_benar = 0;
for k = 1:jumlah_file
    if isequal(hasil_uji{k},kelas_uji{k})
        jumlah_benar = jumlah_benar+1;
    end
end

akurasi_pengujian = jumlah_benar/jumlah_file*100


