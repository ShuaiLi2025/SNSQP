function data = DataSPS(n)

    disp(' Data is generating ...')

    B           = 0.01 * rand(ceil(n/4),n);
    D           = diag(0.01*rand(n,1));
    cov         = B'*B + D;
    r           = 0.5*randn(1,n);
    Qi          = cell(1,1);
    Qi{1}       = 2*D;
    
    data.Q0    = 2*cov;
    data.q0    = zeros(n,1);
    data.Qi    = Qi;
    data.qi    = zeros(n,1);
    data.ci    = -0.001;
    data.ineqA = -r;
    data.ineqb = -0.002;
    data.eqA   = ones(1,n);
    data.eqb   = 1;
    data.lb    = 0;
    data.ub    = 0.3;
    
    disp(' Done data generation !!!')
end
