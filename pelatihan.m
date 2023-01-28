clc; clear; close all; warning off all;

%source nama folder data latih
nama_folder = 'Training';
%membaca file yang berekstensi .jpg
nama_file = dir(fullfile(nama_folder, '*.jpg'));
%membaca jumlah file
jumlah_file = numel(nama_file);

%menginisialisasi variabel

ciri_latih = zeros(jumlah_file,9);

%pengolahan citra terhadap seluruh citra
for n = 1:jumlah_file
    %membaca file citra rgb
    I = imread(fullfile(nama_folder,nama_file(n).name));
    %figure, imshow(I)
    %konversi citra rgb ke citra grayscale
    Imggray = rgb2gray(I);
    %figure, imshow(Imggray)
    %konversi citra grayscale ke citra biner
    bw = imbinarize(Imggray);
    %figure, imshow(bw)
    %operasi komplemen - untuk membalikan nilai piksel
    bw = imcomplement(bw);
    %melakukan operasi morfologi filling holes
    bw = imfill(bw,'holes');
    %figure, imshow(bw)
    %melakukan thresholding 
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
    %mengubah nilai piksel background menjadi nol
    H(~bw) = 0;
    S(~bw) = 0;
    V(~bw) = 0;
    %menghitung rata2 nilai hsv
    Hue = sum(sum(H))/sum(sum(bw));
    Saturation = sum(sum(S))/sum(sum(bw));
    Value = sum(sum(V))/sum(sum(bw));
    Luas = sum(sum(bw));
    
    ciri_latih(n,1) = metric;
    ciri_latih(n,2) = eccentricity;
    ciri_latih(n,3) = Red;
    ciri_latih(n,4) = Green;
    ciri_latih(n,5) = Blue;
    ciri_latih(n,6) = Hue;
    ciri_latih(n,7) = Saturation;
    ciri_latih(n,8) = Value;
    ciri_latih(n,9) = Luas;
end    

%menyusun variabel input
kelas_latih = cell(jumlah_file,1);
%menyusun variabel target

for k = 1:350
    kelas_latih{k} = 'Pineaple';
end
for k = 351:700
    kelas_latih{k} = 'PineapleMini';
end
for k = 701:1050
    kelas_latih{k} = 'Raspberry';
end
for k = 1051:1400
    kelas_latih{k} = 'Redcurrant';
end
for k = 1401:1750
    kelas_latih{k} = 'Strawberry';
end
for k = 1751:2100
    kelas_latih{k} = 'StrawberryWedge';
end
%membangun arsitektur jaringan syaraf tiruan
Knn = fitcknn(ciri_latih,kelas_latih,'NumNeighbors',4);
%membaca nilai keluaran jaringan
hasil_latih = predict(Knn,ciri_latih);


%membaca akurasi
jumlah_benar = 0;
for k = 1:jumlah_file
    if isequal(hasil_latih{k},kelas_latih{k})
        jumlah_benar = jumlah_benar+1;
    end
end

akurasi_pelatihan = jumlah_benar/jumlah_file*100


%menyimpan arsitektur jaringan hasil pelatihan
save Knn Knn