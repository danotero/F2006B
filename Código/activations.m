function a = activations(X,W)

L = size(W,1)+1;
n = size(X,2);
a = cell(L,1);
a{1} = X;

for i = 2:L-1
    a{i} = [ones(1,n);sigmoid(W{i-1}*a{i-1})];
end
    
a{L} = W{L-1}*a{L-1};

end