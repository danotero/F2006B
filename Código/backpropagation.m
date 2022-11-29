function D = backpropagation(L,n,X,W,y)

D = cell(L-1,1);
delta = cell(L,1);
a = activations(X,W);
delta{L} = a{L}-y;
delta{L-1} = (W{L-1}'*delta{L}).*a{L-1}.*(1-a{L-1});

for i = L-2:-1:2
    delta{i} = (W{i}'*delta{i+1}(2:end,:)).*a{i}.*(1-a{i});
end

for i = 1:L-1
    D{i} = (1/n)*delta{i+1}*a{i}';
end

end