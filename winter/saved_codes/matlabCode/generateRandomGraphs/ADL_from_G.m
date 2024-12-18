function [A,D,L] = ADL_from_G(G)
    A = full(adjacency(G,'weighted'));
    D = zeros(size(A));
    for i=1:length(A)
        D(i,i) = sum(A(i,:));
    end
    L = D-A;
end

