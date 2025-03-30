function data = DataRecovery(n,k,m,x,T)
%=========================================================================%
% INPUTS:                                                                 %
%       n  -- size of Q0                                                  %
%       k  -- num of quadratic constraint inequalitys                     %
%       m  -- num of linear constraint inequalitys                        %     
%       x  -- the ground truth                                            %
%       T  -- the support set of xopt                                     %
% OUTPUTS:                                                                %
%       Q0 -- the quadratic objective matrix in R^{n-by-n}                %
%       q0 -- The quadratic objective vector in R^n                       %
%       Qi -- the quadratic constraint matrix in R^{n-by-n}               %
%       qi -- The quadratic constraint vector in R^n                       %
%       ci -- The quadratic constraint constant in R                      %
%       A  -- The linear constraint matrix in R^{m-by-n}                  %
%       b  -- The linear constraint vector in R^m                         %
%-------------------------------------------------------------------------%
    disp(' Data is generating ...')

    l   = ceil(n/4);
    Qi  = cell(k,1);
    qi  = randn(n,k);
    ci  = zeros(k,1);

    B   = randn(n+5,n);
    d   = B(:,T)*x(T);
    Q0  = B'*B;
    q0  = -(d'*B)';

    for i = 1:k
        if i <= ceil(k/2)
            Qii   = randn(l,n);
            Qii   = Qii'*Qii + 0.01*eye(n);
            Qi{i} = Qii;
            ci(i) = -(x(T)'*Qii(T,T)*x(T)/2+qi(T,i)'*x(T))-rand(1);
        else
            Qii   = randn(l,n);
            Qii   = Qii'*Qii + 0.01*eye(n);
            Qi{i} = Qii;
            ci(i) = -(x(T)'*Qii(T,T)*x(T)/2+qi(T,i)'*x(T));
        end
    end
    
    A             = randn(m,n);
    bn            = rand(m,1);
    bn(ceil(m/2)) = 0;
    b             = A*x+bn;

    [~,lambda]    = eigs(Q0,1);
    data.Q0       = Q0/lambda;
    data.q0       = q0/lambda;
    data.Qi       = Qi;
    data.qi       = qi;
    data.ci       = ci;
    data.A        = A;
    data.b        = b;
    
    disp(' Done data generation !!!')
end
