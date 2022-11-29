X = 0:0.01:1;
y = X.^2+0.01*randn(size(X));
X = [ones(1,size(X,2));X];
nodes = [1,128,64,32,1];
L = size(nodes,2);
W = initialize(L,nodes);
alpha = 0.000001;
m = 30000; 
n = size(X,1);
costo = zeros(m,1);

for i = 1:m
    D = backpropagation(L,n,X,W,y);
    for j = 1:L-2
        W{j} = W{j}-alpha*D{j}(2:end,:);
    end
    W{L-1} = W{L-1}-alpha*D{L-1};
    costo(i) = cost(L,n,X,W,y);
end

x = 0:0.0001:1;
y_hat = x.^2;
x = [ones(1,size(x,2));x];
close all
figure
a = activations(x,W);
subplot 121, plot(1:m,costo), axis tight, title('Costo')
subplot 122, plot(x(2,:),a{L}), hold on, plot(x(2,:),y_hat), plot(X(2,:),y)
legend('Prediction','Function','Noisy')

function W = initialize(L,n)

W = cell(L-1,1);

for i = 1:L-1
    W{i} = randn(n(i+1),n(i)+1);
end

end

function c = cost(L,n,X,W,y)

hW = activations(X,W);
c = (1/(2*n))*norm(y-hW{L},'fro')^2;

end