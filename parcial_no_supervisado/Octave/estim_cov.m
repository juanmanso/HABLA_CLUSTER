
function covarianza = estim_cov(x, mu, res)
	n = length(x);
	covarianza = zeros(2,2);

	for i = 1:n
		covarianza = covarianza + (x(i,:)-mu)'*(x(i,:)-mu)*res(i);
	end
	N = sum(res);

	covarianza = covarianza/N;
end
