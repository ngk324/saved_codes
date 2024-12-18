clc; clear all;

counter = 0;
for i = 1:100
    fileB = strcat('results/weights/',num2str(i), '/ObjVal',num2str(i),'.txt');
    fid = fopen(fileB, 'r' ); % changed 
    formatSpec = '%f';
    sizeA = [1 Inf];
    A = fscanf(fid,formatSpec,sizeA);
    fclose(fid);
    if length(A) == 1001
        counter = counter + 1;
    end
    plot(A)
    hold on;
end

%{
edge = zeros(5,1001);
clc; clear all;
for i = 1:154
    fileB = strcat('results/weights/1/lap/',num2str(i),'.txt');
    fid = fopen(fileB, 'r' ); % changed 
    formatSpec = '%f';
    sizeA = [1 Inf];
    A = fscanf(fid,formatSpec,sizeA);
    fclose(fid);
    edge(1,i) = -A(2);
    edge(2,i) = -A(3);
    edge(3,i) = -A(4);
    edge(4,i) = -A(5);
    edge(5,i) = -A(6);
end


for i = 1:5
    plot(edge(i,:))
    hold on;
end
%}