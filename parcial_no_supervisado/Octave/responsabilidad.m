
function [res] = responsabilidad(pos, x, probs, mus, covs)
	num = probs(pos) * mvnpdf(x, mus(pos,:), covs(pos));

	den = 0;
	for i = length(mus)
		den += probs(i) * mvnpdf(x, mus(i), covs(i));
	end

	res = num/den;
end
