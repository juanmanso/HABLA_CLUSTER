
function sigma = estim_sigma(x, mu, N)
	n = length(x);
	sigma = zeros(2,2);
%
%	for i = 1:n
%		if(x(i,:)!= 0)
%			x(k,:)=x(i,:);
%		end
%	end
%	
%	n = k;

	for i = 1:n
		if(x(i,:)!=0)
			sigma = sigma + (x(i,:)-mu)'*(x(i,:)-mu);
		end
	end

	sigma = sigma/N;
end
