function [value,gradient] = oriObjectiveFunction(w,problem)
%% Corrected version of the original objective

h = problem.h;
G = problem.G;
n = numnodes(G);
sEps = problem.stiffnessEps;
gamma = problem.gamma;

G_new = graph(G.Edges);
G_new.Edges.Weight = w(:);
[~,~,L] = generateGraphMatrices(G_new);
[U,diag_lambda] = eig(L);
lambda = diag(diag_lambda);

%% Compute objective value
value = 0;
for i=1:n
    lambda_i = lambda(i);
    for j=1:n
        lambda_j = lambda(j);
        value = value + (h^2 + lambda_i + lambda_j + 2*sEps)/...
            ((lambda_j+sEps)^(2)*(h^4 + 2*h^2*(lambda_i+lambda_j+2*sEps)+(lambda_i-lambda_j)^2));
    end
end
value = h*value/(2*gamma*n^2);

if nargout > 1
    sumOverI = zeros(n,1);
    for l=1:n
        for i=1:n
            if i~=l
                sumElem = (lambda(i)+sEps).^(-2).*((lambda(i)+(-1).*lambda(l)).*(3.*lambda( ...
                    i)+lambda(l)+4.*sEps)+2.*(lambda(i)+(-1).*lambda(l)).*h.^2+(-1).* ...
                    h.^4).*((lambda(i)+(-1).*lambda(l)).^2+2.*(lambda(i)+lambda(l)).* ...
                    h.^2+4.*sEps.*h.^2+h.^4).^(-2)+(lambda(l)+sEps).^(-3).*((lambda(i) ...
                    +(-1).*lambda(l)).^2+2.*(lambda(i)+lambda(l)+2.*sEps).*h.^2+h.^4) ...
                    .^(-2).*((-2).*(lambda(l)+sEps).*((-1).*lambda(i)+lambda(l)+h.^2) ...
                    .*(lambda(i)+lambda(l)+2.*sEps+h.^2)+(lambda(l)+sEps).*((lambda(i) ...
                    +(-1).*lambda(l)).^2+2.*(lambda(i)+lambda(l)+2.*sEps).*h.^2+h.^4)+ ...
                    (-2).*(lambda(i)+lambda(l)+2.*sEps+h.^2).*((lambda(i)+(-1).* ...
                    lambda(l)).^2+2.*(lambda(i)+lambda(l)+2.*sEps).*h.^2+h.^4));
            else
                sumElem = 2.*h.^(-4).*((lambda(l)+sEps).^(-3).*(lambda(l)+sEps+(-1).*h.^2)+( ...
                    -16).*(4.*lambda(l)+4.*sEps+h.^2).^(-2));
            end
            sumOverI(l) = sumOverI(l) + sumElem;
        end
        gradient = zeros(length(w),1);
        edges = G.Edges.EndNodes;
        for i=1:length(edges)
            edge = edges(i,:);
            for k=1:n
                gradient(i) = gradient(i) + sumOverI(k)*(U(edge(1),k)-U(edge(2),k))^2;
            end
        end
        gradient = (h/(2*gamma*n^2))*gradient;
    end
end