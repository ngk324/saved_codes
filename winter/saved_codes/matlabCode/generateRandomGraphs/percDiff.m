clc; clear all;

formatSpec = '%f';
%before = zeros(100,1);
%after = zeros(100,1);
%for i = 9:9
after1 = zeros(100,1);
before1 = zeros(100,1);

fileB = strcat('results/weights_final/ObjAfter.txt');
    fid = fopen(fileB, 'r' ); % changed 
    formatSpec = '%f';
    sizeA = [1 Inf];
    trueAfter = fscanf(fid,formatSpec,sizeA);
    fclose(fid);

    fileB = strcat('results/weights_final/ObjBefore.txt');
    fid = fopen(fileB, 'r' ); % changed 
    formatSpec = '%f';
    sizeA = [1 Inf];
    trueBefore = fscanf(fid,formatSpec,sizeA);
    fclose(fid);

    truePerDecrease = (trueBefore - trueAfter) / trueBefore;

i = 9;
percentDecrease = zeros(100,1);
percentDiff1 = zeros(100,1);
percentDiff2 = zeros(100,1);


for k = 1:100
    %fileB = 'results/weights_final/ObjBefore.txt';
    fileB = strcat('results/weights/',num2str(k),'/ObjBefore',num2str(k),'.txt');
    fid = fopen(fileB, 'r' ); % changed 
    formatSpec = '%f';
    sizeA = [1 Inf];
    before = fscanf(fid,formatSpec,sizeA);
    fclose(fid);


    %fileA = 'results/weights_final/ObjAfter.txt';
    fileA = strcat('results/weights/',num2str(k),'/ObjAfter',num2str(k),'.txt');
    fid = fopen(fileA, 'r' ); % changed 
    formatSpec = '%f';
    sizeA = [1 Inf];
    after = fscanf(fid,formatSpec,sizeA);
    fclose(fid);
    after1(k) = after;
    before1(k) = before;

    
end

    for j = 1:100
        percentDecrease(j) = (before1(j)-after1(j)) / before1(j);
        percentDiff1(j) = abs(trueAfter(j)-after1(j)) / trueAfter(j);
    end

    avgPercentDecrease = mean(percentDecrease);
    avgPercentDiff1 = mean(percentDiff1);

    avgPerDecDiff = mean(abs(truePerDecrease - percentDecrease) / truePerDecrease);


 for k = 1:100
    %fileB = 'results/weights_final/ObjBefore.txt';
    fileB = strcat('results/weights/data_no_convergence/',num2str(k),'/ObjBefore',num2str(k),'.txt');
    fid = fopen(fileB, 'r' ); % changed 
    formatSpec = '%f';
    sizeA = [1 Inf];
    before = fscanf(fid,formatSpec,sizeA);
    fclose(fid);


    %fileA = 'results/weights_final/ObjAfter.txt';
    fileA = strcat('results/weights/data_no_convergence/',num2str(k),'/ObjAfter',num2str(k),'.txt');
    fid = fopen(fileA, 'r' ); % changed 
    formatSpec = '%f';
    sizeA = [1 Inf];
    after = fscanf(fid,formatSpec,sizeA);
    fclose(fid);
    after1(k) = after;
    before1(k) = before;

    
 end


  for j = 1:100
        percentDecrease(j) = (before1(j)-after1(j)) / before1(j);
        percentDiff2(j) = abs(trueAfter(j)-after1(j)) / trueAfter(j);

  end

      avgPerDecDiff2 = mean(abs(truePerDecrease - percentDecrease) / truePerDecrease);

    avgPercentDecreasePrev = mean(percentDecrease);
    avgPercentDiff2 = mean(percentDiff2);



    

    
   