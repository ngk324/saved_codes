function [A,D,L] = generateGraphMatrices(G)
    A = full(adjacency(G,'weighted'));
    D = diag(sum(A,2));
    L = D-A;
end
