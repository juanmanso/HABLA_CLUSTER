%% Función inicializadora de EM y KMEANS

function [m1,m2,m3,cov1,cov2,cov3,prob1,prob2,prob3] = inicializacion(train, indicador, n_clases)

	n2 = n_clases; n3 = 2*n_clases;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Inicialización supervisada %%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if(isnumeric(indicador))	% Si recibo un número que indique cuántos puntos usar:
		m1 = mean(train(1:indicador,:));
		m2 = mean(train(n2+1:(n2+indicador),:));
		m3 = mean(train(n3+1:(n3+indicador),:));

		cov1 = estim_sigma(train(1:indicador,:), m1, indicador);
		cov2 = estim_sigma(train(n2+1:(n2+indicador),:), m2, indicador);
		cov3 = estim_sigma(train(n3+1:(n3+indicador),:), m3, indicador);

		prob1 = 1/3;
		prob2 = 1/3;
		prob3 = 1/3;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Inicialización no supervisada %%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	else
		m1=0; m2=0; m3=0; cov1=0; cov2=0; cov3=0; prob1=0; prob2=0;prob3=0;
		ind = indicador(1);	% Elimino el string para identificar

		% Calculo la media global de los puntos de inicialización
		mu = mean([train(1:ind,:); train(n2+1:(n2+ind),:); train(n3+1:(n3+ind),:)]);
		train1 = train(1:ind,:); train2 = train(n2+1:(n2+ind),:); train3 = train(n3+1:(n3+ind),:);

		% Uniformizo la nube de datos a un circulo
		train = [train1; train2; train3];
		train_sinoff = train-mu;
		mx = max(train_sinoff(:,1)); mix = min(train_sinoff(:,1));
		train_unif(:,1) = train_sinoff(:,1)./(mx-mix)*2;
		mx = max(train_sinoff(:,2)); mix = min(train_sinoff(:,2));
		train_unif(:,2) = train_sinoff(:,2)./(mx-mix)*2;

		[aux, index] = max(vecnorm(train_unif));
		ang_aux = atan2(train_unif(2,index), train_unif(1,index));
		train_unif = [cos(ang_aux) -sin(ang_aux); sin(ang_aux) cos(ang_aux)]*train_unif';
		train_unif = train_unif';

		ang_train = atan2d(train_unif(:,2), train_unif(:,1));

		lim1 = [+90; -30];	% [Límite superior; Límite inf]
		lim2 = [-30; -150];
		lim3 = [+90; -150];

		% Genero los arreglos booleanos
		bool_clase1 = (ang_train<=lim1(1)) & (ang_train>lim1(2));
		bool_clase2 = (ang_train<=lim2(1)) & (ang_train>lim2(2));
		bool_clase3 = (ang_train>=lim3(1)) | (ang_train<lim3(2));

		n1 = sum(bool_clase1(:,1)!=0);
		n2 = sum(bool_clase2(:,1)!=0);
		n3 = sum(bool_clase3(:,1)!=0);

		maxn = 4;

		while((n1 < maxn || n2 < maxn || n3 < maxn) && lim1(1) < 180)
			lim1 += 15;	lim2 += 15;	lim3+= 15;

			bool_clase1 = (ang_train<=lim1(1)) & (ang_train>lim1(2));
			bool_clase2 = (ang_train<=lim2(1)) & (ang_train>lim2(2));
			bool_clase3 = (ang_train>=lim3(1)) | (ang_train<lim3(2));

			n1 = sum(bool_clase1(:,1)!=0);
			n2 = sum(bool_clase2(:,1)!=0);
			n3 = sum(bool_clase3(:,1)!=0);

			if((n1<maxn || n2 <maxn || n3<maxn)&& lim1(1)==180)
				maxn -= 1;
				lim1 -= 90;	lim2 -= 90;	lim3 -= 90;
			end
		end

%			[lim1 lim2 lim3]

		% Separo las muestras con rectas de 120
		clase1 = train.*bool_clase1;
		clase2 = train.*bool_clase2;
		clase3 = train.*bool_clase3;
%		[clase1 clase2 clase3]

%		figure
%		hold on
%		plot(clase1(:,1), clase1(:,2),'r.', 'MarkerSize', 10)
%		plot(clase2(:,1), clase2(:,2),'b.', 'MarkerSize', 10)
%		plot(clase3(:,1), clase3(:,2),'g.', 'MarkerSize', 10)
%		plot(mu(1),mu(2),'kx','MarkerSize',20)


		% Media
		ma = sum(clase1)./sum(clase1(:,1)!=0);
		mo = sum(clase2)./sum(clase2(:,1)!=0);
		mu = sum(clase3)./sum(clase3(:,1)!=0);

		% Corrijo las clases
		[aux, pos_max] = max(([ma(1);mo(1);mu(1)]));
		[aux, pos_min] = max(([ma(1);mo(1);mu(1)]));
%		[aux, pos_max] = max(vecnorm([ma;mo;mu]));
%		[aux, pos_min] = max(vecnorm([ma;mo;mu]));
		aux_clase1=clase1;
		aux_clase2=clase2;
		aux_clase3=clase3;
		if(pos_max==2)
			clase1 = aux_clase2;	m1 = mo;
			if(pos_min==3)
				clase2 = aux_clase1;	m2 = ma;
				clase3 = aux_clase3;	m3 = mu;	%puts('2 1 3');
			else
				clase2 = aux_clase3;	m2 = mu;
				clase3 = aux_clase1;	m3 = ma;	%puts('2 3 1');
			end
		elseif(pos_max==3)
			clase1 = aux_clase3;	m1 = mu;
			if(pos_min==1)
				clase2 = aux_clase2;	m2 = mo;
				clase3 = aux_clase1;	m3 = ma;	%puts('3 2 1');
			else
				clase2 = aux_clase1;	m2 = ma;
				clase3 = aux_clase2;	m3 = mo;	%puts('3 1 2');
			end
		elseif(pos_max==1)
			% Ya está bien
			clase1 = aux_clase1;	m1 = ma;
			if(pos_min==2)
				clase2 = aux_clase3;	m2 = mu;
				clase3 = aux_clase2;	m3 = mo;	%puts('1 3 2');
			else
				% Están bien
				clase2 = aux_clase2;	m2 = mo;
				clase3 = aux_clase3;	m3 = mu;	%puts('1 2 3');
			end
		end


		% Covarianza
		cov1 = estim_sigma(clase1, m1, ind);
		cov2 = estim_sigma(clase2, m2, ind);
		cov3 = estim_sigma(clase3, m3, ind);

		% Probabilidades (ocurrencias/totales)
		prob1 = sum(clase1(:,1)!=0)/length(clase1);
		prob2 = sum(clase2(:,1)!=0)/length(clase2);
		prob3 = sum(clase3(:,1)!=0)/length(clase3);

%		axis tight
%		plot(m1(1),m1(2),'rx','MarkerSize',10)
%		plot(m2(1),m2(2),'bx','MarkerSize',10)
%		plot(m3(1),m3(2),'gx','MarkerSize',10)
	end
end


