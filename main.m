% ***********************************************************************
% Code for the paper:
% Wei Chen, "Simultaneously Sparse and Low-Rank Matrix Reconstruction via Nonconvex and Nonseparable Regularization", IEEE Transactions on Signal Processing (TSP), 66(20), pp.5313-5323, 2018.
%
% Main function
% y: observation vector   y \in R^p
% A: linear mapping matrix  A \in R^{p*nm}
% n,m : dimension of X,   X \in R^(n*m)
% s^2 : sparsity level of X
% r : rank of X
% para.alpha: sparsity-rank trade-off parameter
% para.iters : maximum number of iterations
% para.delta : convergce criteria
% para.lambda : controls variance of Gaussian errors

% ***********************************************************************


n = 50;
m = n;
s = 10;
r = 4;
p = 200;
snr = 1000;

% generate matrix X
X = zeros(n,m);
q1 = randperm(n);
q2 = randperm(m);
Ml = randn(s,r);
Mr = randn(r,s);
X(q1(1:s),q2(1:s)) = Ml*Mr;

% generate sensing matrix A
A = randn(p,n*m);
A = A/norm(A);

% generate y
y = A*X(:);
y = awgn(y,snr,'measured'); 

% normalization
ny = norm(y);
y = y/ny;
X = X/ny;

noise = (norm(y-A*X(:),2)^2)/p;

%============================================begin tests 
% set parameters for converge
para.iters = 500;  % max iter
para.delta = 1e-6;  % convergence control
para.lambda = noise;  % noise parameter
para.alpha = 0.5;

% call algorithm
[hat_X] = SparseLowRankRec(y,A,n,m,para,X);
recovery_error = norm(hat_X-X,'fro')/norm(X,'fro')



