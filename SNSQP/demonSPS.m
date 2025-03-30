% demon SPS problems
clc; clear all; close all; addpath(genpath(pwd));

n             = 1000;
s             = 10;
dt            = DataSPS(n);
pars.x0       = ((dt.lb+dt.ub)/2).*ones(n,1);
pars.tau      = 1; % decrease this value if the algorithm do not converge
pars.dualquad = 0*ones(length(dt.ci));
pars.dualineq = 0.001*ones(length(dt.ineqb)); 
pars.dualeq   = 0.001*ones(length(dt.eqb));
Out           = SNSQP(n,s,dt.Q0,dt.q0,dt.Qi,dt.qi,dt.ci,...
                dt.ineqA,dt.ineqb,dt.eqA,dt.eqb,dt.lb,dt.ub,pars);