clc; clear all;

formatSpec = '%f';
%before = zeros(100,1);
%after = zeros(100,1);
%for i = 9:9
after1 = zeros(100,1);
before1 = zeros(100,1);

i = 9;
percentDiff = zeros(100,1);
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
    fid = fopen('results/AAA-Results3.txt','a');
    varBefore = var(before1);
    varAfter = var(after1);
    avgBefore = mean(before1);
    avgAfter = mean(after1);
    %percentDiff = zeros(length(before),1);
     for j = 1:100
        percentDiff(j) = (before1(j)-after1(j)) / before1(j);
    end

    avgPercentDiff = mean(percentDiff);
    
    fprintf(fid, '\n\n%s\n', strcat('Probability P:   ',num2str(i/20)) );
    fprintf(fid, '\n%s\n', strcat('Initial Avg: ', num2str(avgBefore)) );
    fprintf(fid, '%s\n', strcat('After Avg: ',num2str(avgAfter)) );
    fprintf(fid, '%s\n', strcat('% Decrease Avg: ',num2str(avgPercentDiff)) );
    fprintf(fid, '%s\n', strcat('Initial Var: ',num2str(varBefore)) );
    fprintf(fid, '%s\n', strcat('After Var: ',num2str(varAfter)) );
    
    fclose(fid);
%{
 fid = fopen('results/AAA-Results3.txt','a');
    varBefore = var(before);
    varAfter = var(after);
    avgBefore = mean(before);
    avgAfter = mean(after);
    percentDiff = zeros(length(before),1);
    for j = 1:length(before)-1
        percentDiff(j) = (before(j)-after(j)) / before(j);
    end

    avgPercentDiff = mean(percentDiff);
    
    fprintf(fid, '\n\n%s\n', strcat('Probability P:   ',num2str(i/20)) );
    fprintf(fid, '\n%s\n', strcat('Initial Avg: ', num2str(avgBefore)) );
    fprintf(fid, '%s\n', strcat('After Avg: ',num2str(avgAfter)) );
    fprintf(fid, '%s\n', strcat('% Decrease Avg: ',num2str(avgPercentDiff)) );
    fprintf(fid, '%s\n', strcat('Initial Var: ',num2str(varBefore)) );
    fprintf(fid, '%s\n', strcat('After Var: ',num2str(varAfter)) );
    
    fclose(fid);

%}
    
   