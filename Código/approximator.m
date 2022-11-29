X = 0:0.01:1;
y = X.^2;
% y = X.^2+0.03*randn(size(X));
X = [ones(1,size(X,2));X];
L = 3;
n1 = 1;
n2 = 100;
W = initialize(L,n1,n2);
alpha = 0.00001;
m = 30000; 
n = size(X,2);
costo = zeros(m,1);

for i = 1:m
    D = backprop(L-1,n,X,W,y);
    W{1} = W{1}-alpha*D{1}(2:end,:);
    W{2} = W{2}-alpha*D{2};
    costo(i) = cost(n,X,W,y);
end

x = 0:0.0001:1;
y_hat = x.^2;
x = [ones(1,size(x,2));x];
close all
figure
[~,~,a3] = forward(x,W);
subplot 121, plot(1:m,costo), axis tight, title('Costo')
subplot 122, plot(x(2,:),a3), hold on, plot(x(2,:),y_hat), plot(X(2,:),y)
legend('Prediction','Function','Noisy')

function W = initialize(L,n1,n2)

W = cell(L-1,1);
W{1} = randn(n2,n1+1);
W{2} = randn(1,n2+1);

end

function c = cost(n,X,W,y)

[~,~,hW] = forward(X,W);
c = (1/(2*n))*(y-hW)*(y-hW)';

end

function D = backprop(l,n,X,W,y)

D = cell(l,1);

[a1,a2,a3] = forward(X,W);
delta3 = a3-y;
delta2 = (W{2}'*delta3).*a2.*(1-a2);
D{1} = (1/n)*delta2*a1';
D{2} = (1/n)*delta3*a2';

end