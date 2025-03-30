% demon recovery problems
clc; clear all; close all; addpath(genpath(pwd));
 
n          = 1000; 
k          = ceil(0.01*n); % number of quadratic const
m          = ceil(0.01*n); % number of linear const
s          = ceil(0.05*n);  % sparsity level

test       = 1; % X = R^n        if test =1 
                % X = [-2,2]^n   if test =2 
                % X = [0,inf)^n  if test =3
switch test
  case 1  
       lb   = -inf;
       ub   = inf;         
       xT   = randn(s,1);
  case 2  
       lb   = -2;
       ub   = 2;
       xT   = unifrnd(lb,ub,[s,1]);
  case 3  
       lb   = 0;
       ub   = inf; 
       xT   = 1*rand(s,1); 
end 

T             = randperm(n,s); 
xopt          = zeros(n,1); 
xopt(T)       = xT;
dt            = DataRecovery(n,k,m,xopt,T); 

pars.x0       = zeros(n,1);
pars.tau      = 3; % decrease this value if the algorithm do not converge
pars.dualquad = 0.001*ones(k,1);
pars.dualineq = 0.001*ones(m,1);
pars.itlser   = 1; % increase this value if the algorithm varies a lot
out           = SNSQP(n,s,dt.Q0,dt.q0,dt.Qi,dt.qi,dt.ci,dt.A,dt.b,[],[],lb,ub,pars);
fprintf(' Relerr:   %.4e \n\n', norm(out.sol-xopt)/norm(xopt)); 
PlotRecover(xopt,out.sol,[900,500,500,250],1)