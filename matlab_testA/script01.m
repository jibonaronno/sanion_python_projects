%{function [Cum_conv] = func_conv(u, v)
%    m = length(u);
%    n = length(v);
%
%    for k = 1:n
%        F = 0;
%        for j = max(1, k + 1 - n):min(k, m)
%            F = F + u(j) * v(k + 1 - j);
%        end
%        Cum_conv(k) = F / min(k, m);
%    end
%end
%}

%Test Code
x1=linspace(-2*pi,2*pi); # If vector size is not specified then default size is 100
x2=linspace(-pi,2*pi); # If vector size is not specified then default size is 100
y1=cos(x1);
y2=cos(x2).^2; # Here dot square 2 will square every element in the vector.
plot(x1,y1,'b-',x2,y2,'r:');
title('custom tick');
xlabel('blue:y=cos(x1) where x1=-2*pi,2*pi red:y=cos(x2).^2 where x2=-pi,2*pi');
ylabel('blue:y=cos(x1) red:y=cos(x2).^2');
set(gca, 'XTick', [-2*pi -pi 0 pi 2*pi]);

