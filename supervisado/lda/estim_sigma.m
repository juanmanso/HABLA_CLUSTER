
function sigma = estim_sigma(x, mu)
	n = length(x);
	sigma = zeros(2,2);
	
	for i = 1:n
		sigma = sigma + (x(i,:)-mu)'*(x(i,:)-mu);
	end

	sigma = sigma/n;
end
