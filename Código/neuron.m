x = [1 0 0;1 0 1;1 1 0;1 1 1]';
t = [1, 1, 1, 0]';
w = randn(3,1);
alpha = 0.9;
n = 1000;
costo = zeros(n,1);

for i = 1:n
    z = (w' * x)';
    delta = (sigmoid(z) - t) .* sigmoid_prime(z);
    w = w - alpha * (x * delta);
    costo(i) = cost(t,w,x);
end

x1 = -1:0.1:2;
x2 = -w(1) / w(3) - (w(2) / w(3)) * x1;

close all
figure
subplot 121, plot(1:n,costo), axis tight
subplot 122, hold on, scatter(x(2,:),x(3,:)), plot(x1,x2), ...
    axis tight, xlabel('x_1'), ylabel('x_2')

function c = cost(t,w,x)

z = w' * x;
y = sigmoid(z)';
c = (1/2) * (y - t)' * (y - t);

end