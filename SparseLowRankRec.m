function [X] = SparseLowRankRec(y,A,n,m,para,X0)
% ***********************************************************************
% Code for the paper:
% Wei Chen, "Simultaneously Sparse and Low-Rank Matrix Reconstruction via Nonconvex and Nonseparable Regularization", IEEE Transactions on Signal Processing (TSP), 66(20), pp.5313-5323, 2018.
%
% y: observation vector   y \in R^p
% A: linear mapping matrix  A \in R^{p*nm}
% n,m : dimension of X,   X \in R^(n*m)
% para.iters : maximum number of iterations
% para.delta : convergce criteria
% para.alpha: sparsity-rank trade-off parameter
% para.lambda : controls variance of Gaussian errors
% lambda can be set to 1e-10 for noiseless problems.
% X0: the ground truth matrix. Use it for early stopping in simulations
% ***********************************************************************

lambda = para.lambda;
iters = para.iters;
delta = para.delta;
alpha = para.alpha;

% Initialization
Vx = zeros(n*m,1);
p = length(y);
Psi = eye(n);
X = reshape(Vx,[n,m]);
Cov = sparse(n*m,n*m);
gamma = ones(n,m);

% prepare for the loop
X_old = X;
norml = norm(y);  % to normalize when checking for stop
check = 0;
if nargin == 6
    check = 1;
    normx = norm(X0,'fro');
end


% run
for k = 1:iters
    
    % update X
    for itr=1:m
        [a,b,c] = svd(Psi+diag(gamma(:,itr)));
        temp = length(diag(b)>1e-16);
        temp_inv = c(:,1:temp)*diag(1./(diag(b(1:temp,1:temp))))*a(:,1:temp)';
        temp = diag(gamma(:,itr));
        Cov(((itr-1)*n+1):(itr*n),((itr-1)*n+1):(itr*n)) = temp - temp*temp_inv*temp;
    end
    Vx = Cov*A'*((lambda*sparse(eye(p))+A*Cov*A')\y);
    X = reshape(Vx,[n,m]);
    
    % update Psi
    bar_Psi = sparse(kron(sparse(eye(m)),Psi));
    T = (lambda*sparse(eye(p))+A*bar_Psi*A')\A;
    Psi_sum = 0;
    for i = 1:m
        Ai = A(:,(i-1)*n+1:i*n);
        Ti = T(:,(i-1)*n+1:i*n);
        Ui = Psi - Psi*Ai'*Ti*Psi;
        Psi_sum = Psi_sum + Ui;
    end;
    Psi = Psi_sum/m;
    Psi = Psi + X*(X')./(m - m*alpha);
    
    % update gamma
    G = repmat(sqrt(gamma(:))',p,1);
    PhiG = A.*G;
    [U,S,V] = svd(PhiG,'econ');
    
    diag_S = diag(S);
    U_scaled = U(:,1:p).*repmat((diag_S./(diag_S.^2 + lambda + 1e-16))',p,1);
    Xi = G'.*(V*U_scaled');
    
    PhiGsqr = PhiG.*G;
    Sigma_w_diag = real( gamma(:) - ( sum(Xi.'.*PhiGsqr) ).' );
    gamma = X(:).*X(:)./alpha + Sigma_w_diag;
    gamma = reshape(gamma,[n,m]);
    
    
    % check for stop
    d = norm(X_old-X,'fro')/norml;
    if d < delta
        break;
    end
    X_old = X;    
    
    if check
        % check for stop
        d = norm(X-X0,'fro')/normx;
        if d < 1e-3
            break;
        end
    end
end;


end

