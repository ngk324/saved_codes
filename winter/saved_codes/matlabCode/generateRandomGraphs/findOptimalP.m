clc; clear all;

%numNodes = 112;  % can change # of nodes in graph
%numEdgesPossibile = (numNodes*(numNodes-1))/2;
%probabilities = zeros(1, numNodes);
numProb = 20; % # of probabilities to check, ie 20 means 20 probabilities of 0, 0.05, 0.1 .. to 0.95, 1

for j = 9:9
    for k = 1501:2000 % loops through generating graphs with removing edge with probability j/numProb
        disp(k)
        seed1 = j*k*42;

        rng(seed1);
       folderName = strcat('results/weights/',num2str(k));
       mkdir(folderName);
       randNodeNum = randi(26);
       numNodes = 99+randNodeNum;  % can change # of nodes in graph
       numEdgesPossibile = (numNodes*(numNodes-1))/2;
       probabilities = zeros(1, numNodes);
%{
        fileA = strcat('results/weights_final/',num2str(k),'/ObjAfter.txt');
        fid = fopen(fileA, 'r' ); % changed 
        formatSpec = '%f';
        sizeA = [1 Inf];
        A = fscanf(fid,formatSpec,sizeA);
        fclose(fid);
%}


       % assi       gn probabilities to each node
       for i = 1:numNodes
            prob = rand();
            probabilities(i) = prob;
       end
       disp(j/20);
       newGraph = createGraphByRemoveRand(probabilities,numNodes,j,numProb);
       %plot(newGraph); % if uncommented, loop will end after 1 iteration

       %grad_descent(newGraph,j,k);
       %grad_descent2(newGraph,j,k,A);
       grad_descent2(newGraph,j,k);

    end
end