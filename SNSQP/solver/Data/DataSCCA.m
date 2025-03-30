function data = DataSCCA(nx,ny,N)

    disp(' Data is generating ...')

    v1                    = zeros(nx,1);
    v1(1:nx/8)            = 1;
    v1(nx/8+1:nx/4)       = -1;
    v2                    = zeros(ny,1);
    v2(ny-nx/4+1:ny-nx/8) = 1;
    v2(ny-nx/8+1:end)     = -1;
    u                     = randn(N,1);
    X                     = (v1 + normrnd(0,0.1,nx,1))*u';
    Y                     = (v2 + normrnd(0,0.1,ny,1))*u';
    [a,b,~,~,~]          = canoncorr(X',Y');
    
    Qi      = cell(1,1);
    Qi{1}   = [X*X' zeros(nx,ny); zeros(ny,nx) Y*Y'];

    data.Q0 = [zeros(nx) -X*Y'; -Y*X' zeros(ny)];
    data.q0 = zeros(nx+ny,1);
    data.Qi = Qi;
    data.qi = zeros(nx+ny,1);
    data.ci = -1;
    data.x0 = [a;b];
    
    disp(' Done data generation !!!')
end
