formatSpec = '%f';


%fileB = 'results/weights/1/1_0-3.txt';
%fileB = 'results/weights/1/1_0-8.txt';
%fileB = 'results/weights/1/1_0-9.txt';

fileB = 'results/weights1/2/2_0-3.txt';


%fileB = 'results/weights/1/1_4-37.txt';
sizeA = [1 Inf];
fid = fopen(fileB);
before = fscanf(fid,formatSpec, sizeA);
%close(fid);

plot(before);