clc; clear all;

%% Some parameters
params.h = 0.1;             % Coefficient a of the objective function
params.eps = 10;           % Small dampening added to the laplacian before
                            % computing the eigenvalues
params.gamma = 0.000001;
params.minWeight = 0.001;     % Constraint for weights
params.stiffnessEps = 10;

%% Initialization

for k = 1:100
    disp(k)
    filename = strcat('facebookGraphs/',num2str(k),'.txt');
    fid = fopen(filename, 'r' ); % changed 
    formatSpec = '%d %f';
    sizeA = [1 Inf];
    A = fscanf(fid,formatSpec,sizeA);
    fclose(fid);
    params.nVertices = A(1);
    Adj = zeros(params.nVertices);
    counter = 2;
    for i = 1:params.nVertices
       for j = 1:params.nVertices
           Adj(i,j) = A(counter);
           counter = counter + 1;
       end  
    end 
    
    G_init = graph(Adj);
    G_init = rmedge(G_init, 1:numnodes(G_init), 1:numnodes(G_init));
    
    
    w0 = G_init.Edges(:,2).Variables;
    
    params.G = G_init;
    plot(G_init)
    
    objFuncHandle = @newObjectiveFunction;
    
    disp(['Initial Objective Value: ' ...
        num2str(objFuncHandle(w0,params))])

    % save objective value to file
    fileB = 'facebookResults/FacebookObjBefore.txt';
    fidResultB = fopen(fileB, 'a' );
    fprintf( fidResultB, '%s\n', num2str(objFuncHandle(w0,params)) );
    fclose(fidResultB);
    
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
    
    % save objective value to file
    fileA = 'facebookResults/facebookObjAfter.txt';
    fidResultA = fopen(fileA, 'a' );
    fprintf( fidResultA, '%s\n', num2str(objFuncHandle(w,params)) );
    fclose(fidResultA);
    
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
    fileB = strcat('facebookResults/EVresultsInit',num2str(k),'.txt');
    fileA = strcat('facebookResults/EVresults',num2str(k),'.txt');

    fid = fopen(fileB, 'wt' );
    fid2 = fopen(fileA, 'wt' );
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

    fileB = strcat('facebookResults/LapResultsInit',num2str(k),'.txt');
    fileA = strcat('facebookResults/LapResults',num2str(k),'.txt');
    
    
    fid = fopen(fileB, 'wt' );
    fid2 = fopen(fileA, 'wt' );
    
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
%{
    n_rows = 1;
n_columns = 2;

%% Use the following
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
binWidth = 1.5;
ax0 = nexttile;
maxEV = max(lambda_init) + 1;
hist_init = histogram(lambda_init,0:binWidth:maxEV);
hist_init.LineWidth = 0.1;
maxFreq = max(hist_init.BinCounts) + 5;
ylim([0,maxFreq])
%xlim([0,maxEV])
title("Before optimization", 'FontSize', 8, 'FontWeight', 'bold')
xlabel("Eigenvalue", 'FontSize', 5, 'FontWeight', 'bold')
ylabel("Occurance", 'FontSize', 5, 'FontWeight', 'bold')

ax1 = nexttile;
hold on;
box on;
%maxEV = max(lambda_final);
hist_final = histogram(lambda_final,0:binWidth:maxEV);
hist_final.LineWidth = 0.1;
ylim([0,maxFreq])
%xlim([0,maxEV])
title("After optimization", 'FontSize', 8, 'FontWeight', 'bold')
xlabel("Eigenvalue", 'FontSize', 5, 'FontWeight', 'bold')
ylabel("Occurance", 'FontSize', 5, 'FontWeight', 'bold')

sgtitle("Eigenvalue spectrums", 'FontSize', 8, 'FontWeight', 'bold')
exportgraphics(gcf,'sg_ev_spectrum.pdf','Resolution',600);
%}
%{
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
%}

end