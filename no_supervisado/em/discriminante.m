% Simplificación de la formula completa
% -((x-µ)' inv(cov) (x-µ))/2 - d/2 log(2pi) - 1/2 log(det(cov)) + log(prob)


function discr = discriminante(x, mu, covarianza, prob)
	variable = -1/2 .* (x-mu) * inv(covarianza) *(x - mu)';
	constante = -log(det(covarianza)) + prob;

	discr = variable + constante;
end
