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
input_array = [1 10 2 5 9 100 55 77 0 -5 4 9];
idxA = [1 2 3 4 5 6 7 8 9 10 11 12];

%figure(1);
plot(idxA, input_array, 'k');
title('Useful Signal');
xlabel('Time(s)');
ylabel('Amplitude(u)');
grid on;

