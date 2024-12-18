function [w] = grad_descent2(G_init,prob,num)
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
    
    % form is probFraction-graphNum
    fileB = strcat('results/weights/',num2str(num),'/L_init-',num2str(num),'.txt');
    fidResultB = fopen(fileB, 'w' );
    for i = 1:params.nVertices
        for j = 1:params.nVertices
            fprintf(fidResultB, '%s\n', num2str(L_init(i,j)) );
        end 
    end
       
    fclose(fidResultB);

    folderName = strcat('results/weights/',num2str(num),'/lap');
    mkdir(folderName);
    
    
    w0 = G_init.Edges(:,2).Variables;
    w = w0;
    
    
    objFuncHandle = @newObjectiveFunction;

    fileB = strcat('results/weights/',num2str(num),'/ObjBefore',num2str(num),'.txt');
    fidResultB = fopen(fileB, 'a' );
   
    disp(['Initial Objective Value: ' ...
        num2str(objFuncHandle(w0,params))])

    fprintf( fidResultB, '%s\n', num2str(objFuncHandle(w0,params)) );
    fclose(fidResultB);

    fileB = strcat('results/weights/',num2str(num), '/ObjVal',num2str(num),'.txt');
        fidResultB = fopen(fileB, 'a' );
        fprintf(fidResultB, '%s\n', num2str(objFuncHandle(w0,params)));
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
    options = optimoptions('fmincon','SpecifyObjectiveGradient',false,'MaxIterations',1);

    for j = 1:length(w)
            fileB = strcat('results/weights/',num2str(num), '/', num2str(num),'_', num2str(G_init.Edges.EndNodes(j,1)-1),'-',num2str(G_init.Edges.EndNodes(j,2)-1),'.txt');
            fidResultB = fopen(fileB, 'a' );
            fprintf(fidResultB, '%s\n', num2str(w(j)));
            fclose(fidResultB);
    end

    objval2 = 1000000;

    fileB = strcat('results/weights/',num2str(num),'/lap/1.txt');
    fidResultB = fopen(fileB, 'w' );

    for i = 1:params.nVertices
        for j = 1:params.nVertices
            fprintf( fidResultB, '%s\n', num2str(L_init(i,j)) );
        end 
    end

    fclose(fidResultB);

    % read in obj val

    % loop through 1000 steps of gradient descent

    for i = 1:750
        disp(i)
        %w =
        w1 = fmincon(@(w)objFuncHandle(w,params),w,A,b,Aeq,beq,...
            lb,ub,nonlcon,options);
        objval1 = objFuncHandle(w1,params);
        %perDif = abs((objval1 - finalVal) / objval1);
        if objval1 > objval2 % and within 10% of obj val
            break
        end

        w = w1;
        objval2 = objval1;
        fileB = strcat('results/weights/',num2str(num), '/ObjVal',num2str(num),'.txt');
        fidResultB = fopen(fileB, 'a' );
        fprintf(fidResultB, '%s\n', num2str(objval1));
        fclose(fidResultB);

        for j = 1:length(w)
            % form is probFraction-graphNum_Edge1-Edge2
            fileB = strcat('results/weights/',num2str(num), '/', num2str(num),'_',num2str(G_init.Edges.EndNodes(j,1)-1),'-',num2str(G_init.Edges.EndNodes(j,2)-1),'.txt');
            fidResultB = fopen(fileB, 'a' );
    
            fprintf(fidResultB, '%s\n', num2str(w(j)));
            fclose(fidResultB);
        end


   
    fileB = strcat('results/weights/',num2str(num),'/lap/',num2str(i+1),'.txt');
    fidResultB = fopen(fileB, 'w' );
    
    G_print = G_init;
    G_print.Edges.Weight = w;
    [A_final,~,L_final] = ADL_from_G(G_print);
   
    for k = 1:params.nVertices
        for j = 1:params.nVertices
            fprintf( fidResultB, '%s\n', num2str(L_final(k,j)) );
        end 
    end
       
    fclose(fidResultB);
   
        disp(['Objective Value: ' ...
        num2str(objval1)])
    end

    fileA = strcat('results/weights/',num2str(num),'/ObjAfter',num2str(num),'.txt');

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

    % form is probFraction-graphNum
    fileB = strcat('results/weights/',num2str(num),'/L_Final-',num2str(num),'.txt');
    fidResultB = fopen(fileB, 'w' );

    for i = 1:params.nVertices
        for j = 1:params.nVertices
            fprintf( fidResultB, '%s\n', num2str(L_final(i,j)) );
        end 
    end
       
    fclose(fidResultB);
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
 