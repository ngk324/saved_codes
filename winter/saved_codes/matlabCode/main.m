clc; clear all;

%% Some parameters
nVertices = 10;  % can be changed to adjust # of nodes in graph
params.h = 0.1;             % Coefficient a of the objective function
params.gamma = 0.001;
params.minWeight = 0.1;     % Constraint for weights
params.randEps = 0.3; 
params.stiffnessEps = 1; % Small dampening added to the laplacian before

%% Initialization
w0 = ones(nVertices*(nVertices-1)/2,1);   % Initial guess, same 
                                                        % as the initial state
                                                        % of the graph

% with perturbation
perturbation = -params.randEps + 2*params.randEps*rand(length(w0),1);
w0 = w0 + perturbation;

G = graph();
counter = 1;
for i = 1:nVertices
    for j = i+1:nVertices
        G = addedge(G,i,j,w0(counter));
        counter = counter + 1;
    end
end
params.G = G;
plot(G)

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
options = optimoptions('fmincon','SpecifyObjectiveGradient', ... % MaxFunctionEvaluations and MaxIterations can be changed to ensure convergence
    true,'MaxFunctionEvaluations',1e+5); 

w = fmincon(@(w)objFuncHandle(w,params),w0,A,b,Aeq,beq,...
    lb,ub,nonlcon,options);

disp(['Final Objective Value: ' ...
    num2str(objFuncHandle(w,params))])

%% Post Processing for Generating Histograms
[A_init,D_init,L_init] = generateGraphMatrices(G);
[~,diag_lambda_init] = eig(L_init + params.stiffnessEps*eye(length(L_init)));
lambda_init = diag(diag_lambda_init);

G_new = graph(G.Edges);
G_new.Edges.Weight = w;

[A_final,D_final,L_final] = generateGraphMatrices(G_new);
[~,diag_lambda_final] = eig(L_final + params.stiffnessEps*eye(length(L_final)));
lambda_final = diag(diag_lambda_final);

% saving laplacian matrices and eigenvalues to folder
fid = fopen('EVresultsInit.txt', 'wt' );
fid2 = fopen('EVresults.txt', 'wt' );
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


fid = fopen('LapResultsInit.txt', 'wt' );
fid2 = fopen('LapResults.txt', 'wt' );
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

% plotting EV spectrum for paper
n_rows = 1;
n_columns = 2;

set(0, 'DefaultAxesFontSize', 5); % Must be the size for axis tick labels
set(0, 'DefaultFigureColor', 'w'); % White background
set(0, 'defaulttextinterpreter', 'tex');
set(0, 'DefaultAxesFontName', 'times');
total_figure_width = 8.89;
total_figure_height = 5;
figure
set(gcf, 'Units', 'centimeters');
set(gcf, 'Position', [0 0 total_figure_width total_figure_height]);

tiledlayout(n_rows,n_columns,"TileSpacing","compact")
binWidth = 8;
ax0 = nexttile;
maxEV = max(lambda_final);
hist_init = histogram(lambda_init,0:binWidth:maxEV);
maxFreq = max(hist_init.BinCounts) + 5;
ylim([0,maxFreq])
%xlim([0,maxEV])
title("Before optimization", 'FontSize', 8, 'FontWeight', 'bold')
xlabel("Eigenvalue", 'FontSize', 5, 'FontWeight', 'bold')
ylabel("Occurance", 'FontSize', 5, 'FontWeight', 'bold')

ax1 = nexttile;
hold on;
box on;
hist_final = histogram(lambda_final,0:binWidth:maxEV);
ylim([0,maxFreq])
%xlim([0,maxEV])
title("After optimization", 'FontSize', 8, 'FontWeight', 'bold')
xlabel("Eigenvalue", 'FontSize', 5, 'FontWeight', 'bold')
ylabel("Occurance", 'FontSize', 5, 'FontWeight', 'bold')

sgtitle("Eigenvalue spectrums", 'FontSize', 8, 'FontWeight', 'bold')
exportgraphics(gcf,'cg_ev_spectrum.pdf','Resolution',600);


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