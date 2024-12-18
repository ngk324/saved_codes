clc; clear all;

%% Some parameters  
params.h = 0.1;             % Coefficient a of the objective function
params.gamma = 0.001;
params.minWeight = 0.1;     % Constraint for weights
params.stiffnessEps = 1; % Small dampening added to the laplacian before

%% Initialization

fid = fopen('data/retweet_edges.txt', 'r' ); % changed 
formatSpec = '%d %f';
sizeA = [2 Inf];
A = fscanf(fid,formatSpec,sizeA);
fclose(fid);

x=A(1,:)';
y=A(2,:)';

G_init = graph(x,y,ones(length(x),1));

nVertices = G_init.numnodes;

% if want to make weighted incomplete graph complete
%G_init = convert_to_complete_graph(G_init,params.minWeight); 

w0 = G_init.Edges(:,2).Variables;

params.G = G_init;
plot(G_init)
% with perturbation
%perturbation = -params.randEps + 2*params.randEps*rand(length(w0),1);
%w0 = w0 + perturbation;

objFuncHandle = @newObjectiveFunction;

disp(['Initial Objective Value: ' ...
    num2str(objFuncHandle(w0,params))])

% No linear inequality constraints
A = [];
b = [];

% Weight sum constraint implemented as a linear constraint
Aeq = ones(1,length(w0));
beq = sum(w0);

% It is possible to provide lower and upper bounds for the weights
lb = params.minWeight*ones(length(w0),1);
ub = inf*ones(length(w0),1);

% No nonlinear constraints
nonlcon = [];

% Use options for utilizing the known gradient
options = optimoptions('fmincon','SpecifyObjectiveGradient',true,'MaxFunctionEvaluations',1e+20,'MaxIterations',1e+20);

w = fmincon(@(w)objFuncHandle(w,params),w0,A,b,Aeq,beq,...
    lb,ub,nonlcon,options);

disp(['Final Objective Value: ' ...
    num2str(objFuncHandle(w,params))])

%% Post Processing for Generating Histograms

[A_init,D_init,L_init] = generateGraphMatrices(G_init);
[~,diag_lambda_init] = eig(L_init + params.stiffnessEps*eye(length(L_init)));
lambda_init = diag(diag_lambda_init);

G_new = graph(G_init.Edges);
G_new.Edges.Weight = w;

[A_final,D_final,L_final] = generateGraphMatrices(G_new);
[~,diag_lambda_final] = eig(L_final + params.stiffnessEps*eye(length(L_final)));
lambda_final = diag(diag_lambda_final);

% saving laplacian matrices and eigenvalues to folder
fid = fopen('socialResults/EVresultsInit.txt', 'wt' );
fid2 = fopen('socialResults/EVresults.txt', 'wt' );
for i = 1:length(lambda_init)
    if(i ~= length(lambda_init))
        fprintf( fid, '%f\n', lambda_init(i) );
        fprintf( fid2, '%f\n', lambda_final(i) );
    else
        fprintf( fid, '%f', lambda_init(i) );
        fprintf( fid2, '%f', lambda_final(i) );
    end
end
fclose(fid);
fclose(fid2);


fid = fopen('socialResults/LapResultsInit.txt', 'wt' );
fid2 = fopen('socialResults/LapResults.txt', 'wt' );
for i = 1:length(L_init)
    for j = 1:length(L_init)
        if(i * j ~= length(L_init)*length(L_init))
            fprintf( fid, '%f\n', L_init(i,j) );
            fprintf( fid2, '%f\n', L_final(i,j) );
        else
            fprintf( fid, '%f', L_init(i,j) );
            fprintf( fid2, '%f', L_final(i,j) );
        end
    end
end
fclose(fid);
fclose(fid2);

figure()
subplot(2,2,1)
hist_init = histogram(lambda_init,10);
maxFreq = max(hist_init.BinCounts);
ylim([0,maxFreq])
title("Initial spectrum")
xlabel("Eigenvalue")
ylabel("Frequency")
subplot(2,2,3)
hist_final = histogram(lambda_final,10);
ylim([0,maxFreq])
title("Final spectrum")
xlabel("Eigenvalue")
ylabel("Frequency")

% Plotting graphs
%{
subplot(2,2,2)
max_weight = max(max(A_init,[],"all"),max(A_final,[],"all"));
plotCompleteGraph(A_init,max_weight);
title("Initial graph")
subplot(2,2,4)
plotCompleteGraph(A_final,max_weight);
title("Final graph")

%}