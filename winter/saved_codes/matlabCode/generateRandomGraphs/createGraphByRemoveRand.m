function G = createGraphByRemoveRand(probabilities,numNodes,p,numProb)
    % create complete graph
    G = graph();
    for i=1:numNodes
        for j=i+1:numNodes
            G = addedge(G,i,j,1);
        end
    end
    continueLoop = true;
    % loop until all edges removed while still keeping graph complete
    while continueLoop
        edgeVec = randperm(numedges(G)); % randomly permute order of edges to test if should be removed
        for num = 1:length(edgeVec)
            disp(num)
            edges = G.Edges(edgeVec(num),1).EndNodes;
            i = edges(1);
            j = edges(2);

            if i ~= j % check if edge connecting two nodes should be removed by multiplying prob of nodes
                if probabilities(j) * probabilities(i) < ((p/numProb)) 
                    G = rmedge(G,i,j);
                    % check if removing edge move makes graph incomplete,
                    % if so add edge back to graph
                    [~,~,l] = ADL_from_G(G);
                    e = eig(l);
                    if cast(e(2),"uint8") == 0
                        G = addedge(G,i,j,1);
                    else
                        break
                    end
                end
            end
            % exit once all possible edges removed without making incomplete
            if num == length(edgeVec)
                continueLoop = false;
            end
        end
    end
    [~,order] = sort(degree(G),'descend');
    [G,~] = reordernodes(G,order);
end