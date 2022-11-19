x = [1 -1 1;-1 1 -1]';
[n,m] = size(x);
W = (1/m) * (x * x') - eye(n);
x0 = [-1 -1 -1]';
k = 1000;

for i = 1:k
    y = sign(W * x0);
    x0 = y;
end