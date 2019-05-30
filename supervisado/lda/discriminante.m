
function discr = discriminante(test, mu, Sigma, prob)
	slope = (inv(Sigma)*mu')';
	const = -1/2 .* mu * inv(Sigma) * mu' + prob;

	discr = slope*test' + const;
end
