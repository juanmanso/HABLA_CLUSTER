
function discr = discriminante(test, mu, Sigma, prob)
	var = -1/2 .* (test-mu) * inv(Sigma) *(test - mu)';
	const = -log(det(Sigma)) + prob;

	discr = var + const;
end
