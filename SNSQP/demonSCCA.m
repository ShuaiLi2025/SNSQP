% demon SCCA problems
clc; clear all; close all; addpath(genpath(pwd));

nx            = 200;
ny            = 300;
N             = 50;
s             = 10;
n             = nx + ny;
dt            = DataSCCA(nx,ny,N);
pars.x0       = dt.x0;
pars.tau      = 0.5;  % decrease this value if the algorithm do not converge 
pars.dualquad = 0.01*ones(length(dt.ci));
out           = SNSQP(n,s,dt.Q0,dt.q0,dt.Qi,dt.qi,dt.ci,[],[],[],[],[],[],pars);
fprintf(' Corr:     %.4f \n\n', -out.obj); 
PlotSCCA(out.sol,ceil(nx/200))