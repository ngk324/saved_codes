function [w] = grad_descent(G_init,prob,num)
    %% Some parameters
    params.h = 0.1;             % Coefficient a of the objective function
    params.stiffnessEps = 1;           % Small dampening added to the laplacian before
                                % computing the eigenvalues
    params.gamma = 0.001;
    params.minWeight = 0.1;     % Constraint for weights
    params.randEps = 0.3;

    params.G = G_init;

    params.nVertices = G_init.numnodes; % changed
    
    [A_init,D_init,L_init] = ADL_from_G(G_init);
    [~,diag_lambda_init] = eig(L_init + params.stiffnessEps*eye(length(L_init)));
    lambda_init = diag(diag_lambda_init);
    %{
    % form is probFraction-graphNum
    fileB = strcat('results/L_init-',num2str(prob), '-',num2str(num),'.txt');
    fidResultB = fopen(fileB, 'w' );
    for i = 1:params.nVertices
        for j = 1:params.nVertices
            fprintf(fidResultB, '%s\n', num2str(L_init(i,j)) );
        end 
    end
       
    fclose(fidResultB);
    %}
    
    w0 = G_init.Edges(:,2).Variables;
    
    
    objFuncHandle = @newObjectiveFunction;

    fileB = strcat('results/ObjBefore',num2str(prob),'.txt');
    fidResultB = fopen(fileB, 'a' );
   
    disp(['Initial Objective Value: ' ...
        num2str(objFuncHandle(w0,params))])

    fprintf( fidResultB, '%s\n', num2str(objFuncHandle(w0,params)) );
    fclose(fidResultB);
    
    % No linear inequality constraints
    A = [];
    b = [];
    
    % Weight sum constraint implemented as a linear constraint
    Aeq = ones(1,G_init.numedges);
    beq = sum(w0);
    
    % It is possible to provide lower and upper bounds for the weights
    lb = params.minWeight*ones(G_init.numedges,1);
    ub = inf*ones(G_init.numedges,1);
    
    % No nonlinear constraints
    nonlcon = [];
    
    % Use options for utilizing the known gradient
    options = optimoptions('fmincon','SpecifyObjectiveGradient',true);
    
    w = fmincon(@(w)objFuncHandle(w,params),w0,A,b,Aeq,beq,...
        lb,ub,nonlcon,options);

    fileA = strcat('results/ObjAfter',num2str(prob),'.txt');
    fidResultA = fopen(fileA, 'a' );
    
    disp(['Final Objective Value: ' ...
        num2str(objFuncHandle(w,params))])

    fprintf( fidResultA, '%s\n', num2str(objFuncHandle(w,params)) );
    fclose(fidResultA);

   % [A_final,D_final,L_final] = generateMatricesFromWeights(G_init);
    
    G = G_init;

    G.Edges.Weight = w;
    [A_final,~,L_final] = ADL_from_G(G);
    [U,diag_lambda] = eig(L_final + params.stiffnessEps*eye(length(L_final)));
    lambda_final = diag(diag_lambda);
    %{
    % form is probFraction-graphNum
    fileB = strcat('results/L_Final-',num2str(prob), '-', num2str(num),'.txt');

    fidResultB = fopen(fileB, 'w' );

    for i = 1:params.nVertices
        for j = 1:params.nVertices
            fprintf( fidResultB, '%s\n', num2str(L_final(i,j)) );
        end 
    end
       
    fclose(fidResultB);
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

subplot(2,2,2)
max_weight = max(max(A_init,[],"all"),max(A_final,[],"all"));
plotCompleteGraph(A_init,max_weight);
title("Initial graph")
subplot(2,2,4)
plotCompleteGraph(A_final,max_weight);
title("Final graph")
%}
end
 