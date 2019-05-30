%%%% No puedo pasar así las matrices de cov %%%%

% Recibe la posición de la clase a calcular la responsabilidad como pos
% el vector de probabilidades pi_k como 'prob'
%	idem mu's y covs

% %%%% Se suponen ordenados los vectores %%%% %


function [res] = responsabilidad(pos, x, probs, mus, covs)
	num = probs(pos) * mvnpdf(x, mus(pos,:), covs(pos));

	den = 0;
	for i = length(mus)
		den += probs(i) * mvnpdf(x, mus(i), covs(i));
	end

	res = num/den;
end
