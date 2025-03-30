function Out = SNSQP(n,s,Q0,q0,Qi,qi,ci,ineqA,ineqb,eqA,eqb,lb,ub,pars)

% This code aims at solving the sparse SQCQP in the form of
%
%         min             (1/2)(x'{Q_0}x)+q_0'x, 
%         s.t. (1/2)x'8Qi{i}*x+qi(:,i)'*x+ci(i)<=0, i = 1,...,k,
%                                 ineqA*x-ineqb<=0,
%                                      eqA*x-eqb=0,
%                                        lb<=x<=ub,
%                                       ||x||_0<=s,
% where Qi = {Qi{1},...,Qi{k}}, Qi{i} \in R^{n-by-n}, qi \in R^{n-by-k},  ci \in R^{k}
%       ineqA \in R^{m1-by-n},  ineqb \in R^{m1} 
%       eqA   \in R^{m2-by-n},  eqb   \in R^{m2}
%       s << n

%---------------------------------------------------------------------------------------------------           
% Inputs:
%     n:      Dimension of the solution x                                               (required)
%     s:      Sparsity level of x, an integer between 1 and n-1                         (required)
%     Q0:     The quadratic objective matrix in R^{n-by-n}                              (required)        
%     q0:     The quadratic objective vector in R^n                                     (required)
%     Qi:     The quadratic constraint matrix   
%             MUST be a cell array or [], each entry is matrix in R^{n-by-n}            (optional)
%     qi:     The quadratic constraint vector. MUST be a matrix in R^{n-by-k} or []     (optional)           
%     ci:     The quadratic constraint constant in R, must be a vector or []            (optional)
%     ineqA:  The linear inequality constraint matrix in R^{m1-by-n}   or []            (optional)
%     ineqb:  The linear inequality constraint vector in R^{m1}        or []            (optional)
%     eqA:    The linear equality constraint matrix in R^{m2-by-n}     or []            (optional)
%     eqb:    The linear equality constraint vector in R^{m2}          or []            (optional)
%     lb:     The lower bound of x                                                      (optional)
%     ub:     The upper bound of x                                                      (optional)
%             NOTE: 0 must in [lb ub]
%     pars:   Parameters are all OPTIONAL
%             pars.x0       -- Starting point of x                                      (default zeros(n,1))
%             pars.dualquad -- Starting point of mu for quadratic constraint            (default zeros(k,1))
%             pars.dualineq -- Starting point of lamb1 for linear inequality constraint (default zeros(m1,1))
%             pars.dualeq   -- Starting point of lamb2 for linear equality constraint   (default zeros(m2,1))
%             pars.dualbd   -- Starting point of nu  for bound/box constraint           (default zeros(n,1))
%             pars.tau      -- A positive scalar                                        (default 1)
%                              NOTE: tuning a proper tau may result in better solutions     
%             pars.itlser   -- Maximum nonumber of line search                          (default 5)
%             pars.itmax    -- Maximum nonumber of iteration                            (default 10000)
%             pars.show     -- Results shown at each iteration if pars.show=1           (default 1)
%                              Results no shown at each iteration if pars.show=0
%             pars.tol      -- Tolerance of the halting condition,                      (default 1e-6)
%
% Outputs:
%     Out.sol:           The sparse solution x
%     Out.sparsity:      Sparsity level of Out.sol
%     Out.error:         Error used to terminate this solver
%     Out.time           CPU time
%     Out.iter:          Number of iterations
%     Out.obj:           Objective function value at Out.sol
%---------------------------------------------------------------------------------------------------
% This code is programmed based on the algorithm proposed in 
% "Shuai Li, Shenglong Zhou, and Ziyan Luo, Sparse quadratically constrained
%  quadratic programming via semismooth Newton method, arXiv:2503.15109,2025." 
%  Send your comments and suggestions to <<< 24110488@bjtu.edu.cn / slzhou2021@163.com >>>
%  Warning: Accuracy may not be guaranteed !!!!! 
%---------------------------------------------------------------------------------------------------

warning off;
t0     = tic;

if  nargin < 13
    disp(' No enough inputs. No problems will be solverd!'); return;
end
if nargin < 14; pars = [];  end 
 
[dim,existcons,flagbd,lenf,show,x0,dualquad,dualineq,dualeq,dualbd,tau,tol,itmax,...
itlser,gamma,sigma,alpha0,lb,ub] = setparameters(n,s,Qi,ineqA,eqA,lb,ub,pars);

% The main body
if  show
    fprintf('\n Start to run the sover -- SNSQP\n');
    fprintf(' -------------------------------------------------\n');
    fprintf(' Iter        Error        Objective      Time(sec)\n');
    fprintf(' -------------------------------------------------\n');
end

%--------------------------------Initialization---------------------------
Fnorm    = @(var)norm(var,'fro')^2;
Ffun     = @(x,T)FuncObj(x,Q0,q0,T);
FQxq     = @(x,T)FuncQxq(x,Qi,qi,dim(1),T);
FQxqc    = @(x,T)FuncQxqc(x,Qi,qi,ci,dim(1),T);
FIneqAxb = @(x,T)FuncAxb(x,ineqA,ineqb,T);
FEqAxb   = @(x,T)FuncAxb(x,eqA,eqb,T);
FGradL   = @(xT,T,vq,vi,ve,Qxq)GradLag(xT,Q0,q0,Qxq,ineqA,eqA,vq,vi,ve,T,existcons);
FHessL   = @(vq,T)HessLagT(Q0,Qi,vq,T,dim(1));
FHessLc  = @(vq,T,TTc)HessLagTc(Q0,Qi,vq,T,TTc,dim(1));

if  existcons(4)
    FProjbd = @(xT)Projbd(xT,lb,ub);
    JProjbd = @(xT)JacProjbd(xT,lb,ub);
end

z       = zeros(n,1);
Index   = 1:n;
tau0    = tau;
x       = x0;
[~, T0] = maxk(x,s,'ComparisonMethod','abs');
xT      = x(T0);
obj     = Ffun(xT,T0);
if  existcons(1)
    Qxq     = FQxq(xT,T0);
    Qqc     = FQxqc(xT,T0);
    Ncpqual = FuncNcpduad(Qqc,dualquad);
else
    Qxq     = [];
    Ncpqual = [];
end
if  existcons(2)
    Axb     = FIneqAxb(xT,T0);
    Ncpineq = FuncNcpineq(Axb,dualineq);
else
    Ncpineq = [];
end
if  existcons(3)
    Lineq   = FEqAxb(xT,T0);
else
    Lineq   = [];
end
if  existcons(4)
    dualbdT   = dualbd(T0);  
    xPT       = xT - FProjbd(xT+dualbdT);
    GradL     = FGradL(xT,T0,dualquad,dualineq,dualeq,Qxq);
    GradL(T0) = GradL(T0)+dualbdT;
else
    xPT       = [];
    GradL     = FGradL(xT,T0,dualquad,dualineq,dualeq,Qxq);
end

Indx    = 1:s ;
Indqual = s+1:s+dim(1);
Indineq = s+dim(1)+1:s+sum(dim(1:2));
Indeq   = s+sum(dim(1:2))+1:s+sum(dim);
   
for iter = 1:itmax    
    %--------------------------------Index selection---------------------------
    [~, T]     = maxk(x-tau*GradL,s,'ComparisonMethod','abs');
    Tc         = Index;
    Tc(T)      = [];
    TTc        = setdiff(T0,T);
    flagT      = isempty(TTc);

    %--------------------------------Stop criterion----------------------------
    if existcons(4)
       if ~flagT
           xPT = x(T) - FProjbd( x(T)+dualbd(T) );
       end
    end
    StationEq = [GradL(T); Ncpqual; Ncpineq; Lineq; xPT];
    error     = norm(StationEq) + norm(x(Tc));

    if show 
        fprintf('%4d       %5.2e      %10.3e    %7.3fsec\n',iter,error,obj,toc(t0));
    end

    if error<=tol; break;  end

    %----------------------------- Newton step---------------------------------
    HessTT  = FHessL(dualquad,T);
    if existcons(1)
        QxqT   = Qxq(T,:);
        [JPquad, JDquad] = JacNcpquad(Qqc,dualquad,dim(1));
        Hquad  =  JPquad.*(QxqT)';
    else
        QxqT   = [];
        JPquad = [];
        JDquad = [];
        Hquad  = [];
    end
    
    if existcons(2)
        IneqAT = ineqA(:,T);
        [JPineq, JDineq] = JacNcpineq(Axb,dualineq,dim(2));
        Hineq  = JPineq.*IneqAT;
    else
        IneqAT = [];
        JPineq = [];
        JDineq = [];
        Hineq  = [];
    end
    
    if existcons(3)
        EqAT   = eqA(:,T);
    else
        EqAT   = [];
    end
    
    if existcons(4)
        if flagT
            projbd = JProjbd(xT+dualbdT);
        else
            projbd = JProjbd(x(T)+dualbd(T));
        end
        U   = diag(projbd);
        eU  = diag(1-projbd);
        sbd = s;
    else
        U   = [];
        eU  = [];
        sbd = 0;
    end
    
    HessL  = [HessTT QxqT IneqAT' EqAT' eye(sbd);
              Hquad diag(JDquad) zeros(dim(1),dim(2)+dim(3)+sbd);
              Hineq zeros(dim(2),dim(1)) diag(JDineq) zeros(dim(2),dim(3)+sbd);
              EqAT zeros(dim(3),sum(dim)+sbd);
              eU zeros(sbd,sum(dim)) -U];
        
    if iter == 1 || flagT || flagbd    % update next iterate if T==supp(x^k)
        STEq   = -StationEq;       
    else                               % update next iterate if T~=supp(x^k)
        HessTc = FHessLc(dualquad,T,TTc);
        if  existcons(1)
            STcquad = JPquad.*(Qxq(TTc,:))';
        else
            STcquad = [];
        end
        
        if  existcons(2)
            STcineq = JPineq.*ineqA(:,TTc);
        else
            STcineq = [];
        end
        
        if  existcons(3)
            STceq   = eqA(:,TTc);
        else
            STceq   = [];
        end
        
        if  existcons(4)
            STcbd   = zeros(s,length(TTc));
        else
            STcbd   = [];
        end
        STEq  = - StationEq + ([HessTc;STcquad;STcineq;STceq;STcbd]*x(TTc));
    end
    
    if s   < 1000
        d  = HessL\STEq;
    else
        d  = my_cg(HessL,STEq,1e-16,20,zeros(lenf,1)); 
    end

    if max(isnan(d))==1
       d    = (HessL'*HessL + (0.01/iter)*speye(lenf))\(STEq'*HessL)';
    end

    %----------------------------- Line search---------------------------------                
    mark = 0;
    while 1
        xT1         = x(T)     + d(Indx);
        dualquad1   = dualquad + d(Indqual);
        dualineq1   = dualineq + d(Indineq);
        dualeq1     = dualeq   + d(Indeq);
        
        if  existcons(1)
            Qxq     = FQxq(xT1,T);
            Qqc     = FQxqc(xT1,T);
            Ncpqual = FuncNcpduad(Qqc,dualquad1);
        else
            Ncpqual = [];
        end
        
        if  existcons(2)
            Axb     = FIneqAxb(xT1,T);
            Ncpineq = FuncNcpineq(Axb,dualineq1);
        else
            Ncpineq = [];
        end
        
        if  existcons(3)
            Lineq  = FEqAxb(xT1,T);
        else
            Lineq  = [];
        end
        
        if  existcons(4)
            dualbdT1  = dualbd(T) + d(s+sum(dim)+1:end);
            xPT       = xT1 - FProjbd(xT1+dualbdT1);
            gradL1    = FGradL(xT1,T,dualquad1,dualineq1,dualeq1,Qxq);
            gradL1(T) = gradL1(T)+dualbdT1;
        else
            gradL1    = FGradL(xT1,T,dualquad1,dualineq1,dualeq1,Qxq);
        end
        F1            = [gradL1(T); Ncpqual; Ncpineq; Lineq; xPT];
        
        if  norm(F1) < 1e4*error || mark==2
            break;
        elseif mark ~= 1
            mark     = 1;
            d        = (HessL'*HessL + (0.01/iter)*speye(lenf))\(STEq'*HessL)';
        else
            mark     = 2;
            d        = STEq;    
        end
    end

    alpha = alpha0;
    tmp   = Fnorm(StationEq)+Fnorm(x(Tc));
    for j = 1:itlser
        if Fnorm(F1) < (1 - 2*sigma*alpha)*tmp;  break; end
        alpha     = alpha*gamma;
        xT1       = x(T)     + alpha*d(Indx);
        dualquad1 = dualquad + alpha*d(Indqual);
        dualineq1 = dualineq + alpha*d(Indineq);
        dualeq1   = dualeq   + alpha*d(Indeq);
        
        if  existcons(1)
            Qxq     = FQxq(xT1,T);
            Qqc     = FQxqc(xT1,T);
            Ncpqual = FuncNcpduad(Qqc,dualquad1);
        else
            Ncpqual = [];
        end
        
        if  existcons(2)
            Axb     = FIneqAxb(xT1,T);
            Ncpineq = FuncNcpineq(Axb,dualineq1);
        else
            Ncpineq = [];
        end
        
        if  existcons(3)
            Lineq   = FEqAxb(xT1,T);
        else
            Lineq   = [];
        end
        
        if existcons(4)
            dualbdT1  = dualbd(T) + alpha*d(s+sum(dim)+1:end);
            P         = FProjbd(xT1+dualbdT1);
            xPT       = xT1 - P;
            gradL1    = FGradL(xT1,T,dualquad1,dualineq1,dualeq1,Qxq);
            gradL1(T) = gradL1(T)+dualbdT1;
        else
            gradL1    = FGradL(xT1,T,dualquad1,dualineq1,dualeq1,Qxq);
        end
        
        F1            = [gradL1(T); Ncpqual; Ncpineq; Lineq; xPT];
    end

    obj       = Ffun(xT1,T);
    x         = z;
    xT        = xT1;
    x(T)      = xT1; 
    dualquad  = dualquad1;
    dualineq  = dualineq1;
    dualeq    = dualeq1;
    if existcons(4)
        dualbd    = z;
        dualbdT   = dualbdT1;
        dualbd(T) = dualbdT1;
    end
    GradL        = gradL1;
    T0           = T;    
    if mod(iter,200)==0; tau = max(0.1*tau0, tau/1.5); end
    
end



% results output
time             = toc(t0);
Out.sol          = x;
Out.sparsity     = nnz(x); 
Out.error        = error;
Out.time         = time;
Out.iter         = iter;
Out.obj          = obj;

if show
    fprintf(' -------------------------------------------------\n');
    fprintf(' Obj :     %.4f\n', Out.obj );
    fprintf(' Time:     %.4f seconds\n', Out.time);
    %fprintf(' Iter:    %5d \n', Out.iter);
end

end


% Set up parameters--------------------------------------------------------
function [dim,existcons,flagbd,lenf,show,x0,dualquad,dualineq,dualeq,dualbd,...
          tau,tol,itmax,itlser,gamma,sigma,alpha0,lb,ub]= setparameters(n,s,Qi,ineqA,eqA,lb,ub,pars)
    
dim          = [length(Qi) size(ineqA,1) size(eqA,1) ];  
existcons    = ones(4,1);
existcons(1) = dim(1)>0;
existcons(2) = dim(2)>0;
existcons(3) = dim(3)>0;
if isempty(lb); lb = -inf; end
if isempty(ub); ub =  inf; end 
existcons(4) =  1-( lb==-inf & ub==inf );  
lenf         = s+sum(dim)+s*existcons(4);  
flagbd       = ( lb==0 | ub==0 );

if isfield(pars,'show');   show   = pars.show;   else; show   = 1;          end     
if isfield(pars,'itmax');  itmax  = pars.itmax;  else; itmax  = 1e4;        end
if isfield(pars,'x0');     x0     = pars.x0;     else; x0     = zeros(n,1); end
if isfield(pars,'tau');    tau    = pars.tau;    else; tau    = 1;          end
if isfield(pars,'tol');    tol    = pars.tol;    else; tol    = 1e-6;       end
if isfield(pars,'itlser'); itlser = pars.itlser; else; itlser = 5;          end
if isfield(pars,'gamma');  gamma  = pars.gamma;  else; gamma  = 0.5;        end
if isfield(pars,'sigma');  sigma  = pars.sigma;  else; sigma  = 1e-4;       end
if isfield(pars,'alpha0'); alpha0 = pars.alpha0; else; alpha0 = 1;          end  

if isfield(pars,'dualquad') && length(pars.dualquad)==dim(1)     
    dualquad    = pars.dualquad;    
else 
    dualquad    = zeros(dim(1),1); 
end

if isfield(pars,'dualineq') && length(pars.dualineq)==dim(2)  
    dualineq = pars.dualineq; 
else
    dualineq = zeros(dim(2),1);      
end

if isfield(pars,'dualeq') && length(pars.dualeq)==dim(3)  
    dualeq = pars.dualeq; 
else  
    dualeq = zeros(dim(3),1);      
end

if isfield(pars,'dualbd') && length(pars.dualbd)==n && existcons(4)   
    dualbd = pars.dualbd;   
else
    dualbd = zeros(n*existcons(4),1);    
end
    
end


% Objection ---------------------------------------------------------------
function f = FuncObj(xT,Q0,q0,T)
         f = (1/2)*xT'*Q0(T,T)*xT+sum(q0(T).*xT);
end

% Gradient of Lagragian function ------------------------------------------
function g = GradLag(xT,Q0,q0,Qxq,IneqA,EqA,mu,lamb1,lamb2,T,iscons)
    g      = Q0(:,T)*xT+q0;
    if iscons(3)
        g  = g + (lamb2'*EqA)';
    end
    if iscons(2)
        g  = g + (lamb1'*IneqA)';
    end
    if iscons(1)
        g  = g + Qxq*mu;
    end
end

% HessianLTT --------------------------------------------------------------
function h = HessLagT(Q0,Qi,mu,T,k)
    h      = Q0(T,T);
    for i  = 1:k 
        h  = h + mu(i)*Qi{i}(T,T);
    end
    h      = double(h);
end

% HessianLTTc--------------------------------------------------------------
function hc = HessLagTc(Q0,Qi,mu,T,TTc,k)
    hc      = Q0(T,TTc);
    for i   = 1:k
        hc  = hc + mu(i)*Qi{i}(T,TTc);
    end
    hc      = double(hc);
end

% Ax-b---------------------------------------------------------------------
function Axb = FuncAxb(xT,A,b,T)
    Axb      = -b;
    if ~isempty(T)
        Axb  = Axb +A(:,T)*xT;
    end
end

% (Qx+q)-------------------------------------------------------------------
function Qxq = FuncQxq(xT,Qi,qi,k,T)
    Qxq        = qi;
    if ~isempty(T) 
        for i = 1:k
            Qxq(:,i) = Qxq(:,i) + Qi{i}(:,T)*xT ;
        end 
    end
end

% (1/2)x'Qx+qx+c-----------------------------------------------------------
function Qqc = FuncQxqc(xT,Qi,qi,ci,k,T)
    Qqc = ci;
    if ~isempty(T)  
        tmp = xT'*qi(T,:);
        for i = 1:k
            Qqc(i) = xT'*Qi{i}(T,T)*xT/2 + tmp(i) +  Qqc(i)   ;
        end
    end      
end

% Ncp of quadrtic cons-----------------------------------------------------
function h1 = FuncNcpduad(Qqc,dualquad)
    h1 = sqrt(Qqc.^2+dualquad.^2) + Qqc - dualquad;
end

% Ncp for ineaqulity cons-------------------------------------------------- 
function h2 = FuncNcpineq(Axb,dualineq)
    h2 = sqrt(Axb.^2+dualineq.^2) + Axb - dualineq;
end

%Jacobian of Ncp for quadrtic cons----------------------------------------
function [jprimquad, jdualquad] = JacNcpquad(Qqc,dualquad,k)
    zsqrt     = sqrt(Qqc.^2+dualquad.^2);  
    jprimquad = Qqc./zsqrt+1;
    jdualquad = dualquad./zsqrt-1;
    for i = 1:k
        if zsqrt(i) == 0          
            angle    = rand(1)*2*pi;
            r        = sqrt(rand(1));
            jprimquad(i) = r*cos(angle)+1;
            jdualquad(i) = r*sin(angle)-1;
        end
    end
end

%Jacobian of Ncp for inequality cons----------------------------------------
function [jprimineq, jdualineq] = JacNcpineq(Axb,dualineq,m)
    zsqrt     = sqrt(Axb.^2+dualineq.^2);
    jprimineq = Axb./zsqrt+1;
    jdualineq = dualineq./zsqrt-1;
    for i = 1:m
        if zsqrt(i) == 0 
            angle    = rand(1)*2*pi;
            r        = sqrt(rand(1));
            jprimineq(i) = r*cos(angle)+1;
            jdualineq(i) = r*sin(angle)-1;
        end
    end
end

% projection operator of box constrait-------------------------------------
function pbd = Projbd(xT,lb,rb)
   pbd = min(max(xT,lb),rb);
end

% Clarke Jacobian of projection operator--------------------------
function jpbd = JacProjbd(xT,lb,ub)
    sx   = length(xT);
    jpbd = zeros(sx,1);
    jpbd( xT> lb & xT< ub ) = 1;
    jpbd( xT==lb | xT==ub ) = 1/2;    
end

% Conjugate gradient descent to solve linear equations--------------------- 
function x = my_cg(fx,b,cgtol,cgit,x)
    r = b;
    e = norm(r,'fro')^2;
    t = e;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        if  i == 1  
            p = r;
        else
            p = r + (e/e0)*p;
        end  
        if  isa(fx,'function_handle')
            w  = fx(p);
        else
            w  = fx*p;
        end
        a  = e/sum(p.*w);
        x  = x + a * p; 
        r  = r - a * w;  
        e0 = e;
        e  = norm(r,'fro')^2;
    end 
  
end




