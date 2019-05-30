%% EM

% Clasificar:
% p(clases=c | x) = ( p(x|clase=c) * P(clase=c) ) / p(x)

% Paso E:
% \gamma_c (x) = ( p(x|clase=c; \theta^{n-1}) * P(c;\theta^{n-1}) ) / p(x;\theta^{n-1})

% Paso M:
% \µ^n_c = \sum x * \gamma_c
% cov^n_c = (\sum (x-µ^n_c)*(x-µ^n_c)T * \gamma_c (x)) / sum(\gamma_c(x))
% pi_c = P(c; \theta^{n-1}) = sum(\gamma_c(x))/N


clear all
close all
%graphics_toolkit('gnuplot');
myGreen = [0 0.5 0];

a_total = load('a.txt');
o_total = load('o.txt');
u_total = load('u.txt');

%% Separo en test y train
% a
ind_perm = randperm(length(a_total));	% Entreno en cada corrida con otras muestras
a_train = a_total(ind_perm(1:35),1:2);
a_test = a_total(ind_perm(36:end),1:2);
% o
ind_perm = randperm(length(o_total));	% Entreno en cada corrida con otras muestras
o_train = o_total(ind_perm(1:35),1:2);
o_test = o_total(ind_perm(36:end),1:2);
% u
ind_perm = randperm(length(u_total));	% Entreno en cada corrida con otras muestras
u_train = u_total(ind_perm(1:35),1:2);
u_test = u_total(ind_perm(36:end),1:2);

%% Gráficos conjunto de puntos
train = [a_train;o_train;u_train];

% Hago el 'scatter' de las muestras sin clasificar
figure
hold on
plot(train(:,1), train(:,2),'k.', 'MarkerSize', 10)
grid minor
close

%% Entrenamiento:
%1) inicialización con puntos: usar 5 puntos de train con etiquetas para inicializar la media:
ma_inic = mean(a_train(1:5,:));
mo_inic = mean(o_train(1:5,:));
mu_inic = mean(u_train(1:5,:));

%2) inicialización con puntos para sigma: (igual que lda)
cova_inic = estim_sigma(a_train(1:5,:), ma_inic, 5);
covo_inic = estim_sigma(o_train(1:5,:), mo_inic, 5);
covu_inic = estim_sigma(u_train(1:5,:), mu_inic, 5);
cov_general_inic = (cova_inic + covo_inic + covu_inic)/3;

%3) inicialización con puntos para la probabilidad:
proba_inic = 1/3;
probo_inic = 1/3;
probu_inic = 1/3;

%% iteraciones:
% Medias, covarianzas y pi's de las clases
ma = ma_inic;	cova = cov_general_inic;	proba = 1/3;
mo = mo_inic;	covo = cova;	probo = 1/3;
mu = mu_inic;	covu = cova;	probu = 1/3;

% Label
real_label = [ones(length(a_train),1);2*ones(length(o_train),1);3*ones(length(u_train),1)];

likelihood(2) = -10;
likelihood(1) = -10.01;
limlike = 0.001;
n = 2;

%% Preparo el gráfico
theta = linspace(0, 2*pi, 100);
rot = [sin(theta); cos(theta)];

figure(1); hold on;
% Medias con X
plot(ma(1),ma(2), 'rx','MarkerSize',10);
plot(mo(1),mo(2),'x','color',myGreen,'MarkerSize',10);
plot(mu(1),mu(2), 'bx','MarkerSize',10);
% Puntos bien clasificados
plot(train(:,1).*(real_label==1), train(:,2).*(real_label==1), 'r.', 'MarkerSize',10);
plot(train(:,1).*(real_label==2), train(:,2).*(real_label==2), '.', 'color', myGreen, 'MarkerSize',10);
plot(train(:,1).*(real_label==3), train(:,2).*(real_label==3), 'b.', 'MarkerSize',10);
% ELIPSES
elipsea = (chol(cova)' * rot)' + ma;
elipseo = (chol(covo)' * rot)' + mo;
elipseu = (chol(covu)' * rot)' + mu;
plot(elipsea(:,1),elipsea(:,2),'r');
plot(elipseo(:,1),elipseo(:,2),'color', myGreen);
plot(elipseu(:,1),elipseu(:,2),'b');
axis tight

titles = ['Iter0'; 'Iter1'; 'Iter2'; 'Iter3'; 'Iter4'; 'Iter5'; 'Iter6'; 'Iter7'; 'Iter8'; 'Iter9'; 'Iter10'; 'Iter11'; 'Iter12'; 'Iter13'; 'Iter14'; 'Iter15'; 'Iter16'; 'Iter17'; 'Iter18'; 'Iter19'];
title(titles(1,:));

%% Iteración

bool_plot = 0;
N = length(train);

while( abs((likelihood(n-1) - likelihood(n))) > limlike && n<20)
	% %%% PASO E %%% %
	% Calculo de responsabilidades para cada muestra
	for i = 1:length(train)
		x = train(i,:);
		res(1) = mvnpdf(x, ma, cova)*proba;
		res(2) = mvnpdf(x, mo, covo)*probo;
		res(3) = mvnpdf(x, mu, covu)*probu;
		den = sum(res);

		gama(i,:) = res/den;
	end

	% %%% PASO M %%% %
	NA = sum(gama(:,1));	NO = sum(gama(:,2));	NU = sum(gama(:,3));
	% Recalculo:
	% media
	ma = sum(train.*gama(:,1))/sum(gama(:,1));
	mo = sum(train.*gama(:,2))/sum(gama(:,2));
	mu = sum(train.*gama(:,3))/sum(gama(:,3));

	% covarianza
	cova = estim_cov(train, ma, gama(:,1));
	covo = estim_cov(train, mo, gama(:,2));
	covu = estim_cov(train, mu, gama(:,3));

	% pi (probabilidad)
	proba = sum(gama(:,1))/sum(sum(gama));
	probo = NO/N;
	probu = NU/N;

	% %%% LIKELIHOOD %%% %	optimizar
	den = 0;
	for i = 1:length(train)
		x = train(i,:);
		res(1) = mvnpdf(x, ma, cova)*proba;
		res(2) = mvnpdf(x, mo, covo)*probo;
		res(3) = mvnpdf(x, mu, covu)*probu;
		den += sum(res);
	end

	%log_likelihood
	likelihood(n+1) = sum(log(den));

	if(bool_plot)
		% Gráfico
		figure(n); hold on;
		% Medias con X
		plot(mu(1),mu(2), 'bx','MarkerSize',10);
		plot(ma(1),ma(2), 'rx','MarkerSize',10);
		plot(mo(1),mo(2),'x','color',myGreen,'MarkerSize',10);
		% Puntos bien clasificados
		plot(train(:,1).*(real_label==3), train(:,2).*(real_label==3), 'b.', 'MarkerSize',10);
		plot(train(:,1).*(real_label==1), train(:,2).*(real_label==1), 'r.', 'MarkerSize',10);
		plot(train(:,1).*(real_label==2), train(:,2).*(real_label==2), '.', 'color', myGreen, 'MarkerSize',10);
		% Puntos clasificados
		scatter(train(:,1), train(:,2), 10, gama);
		% ELIPSES
		elipsea = (chol(cova)' * rot)' + ma;
		elipseo = (chol(covo)' * rot)' + mo;
		elipseu = (chol(covu)' * rot)' + mu;
		plot(elipseu(:,1),elipseu(:,2),'b');
		plot(elipsea(:,1),elipsea(:,2),'r');
		plot(elipseo(:,1),elipseo(:,2),'color', myGreen);

		axis tight
		title(titles(n,:));
	end

	n += 1;
end


%% Plots de datos tras entrenamiento
% Es el último gráfico de la iteración

%%% Plot likelihood
figure
plot(likelihood);
title('Logaritmo de la verosimilitud en función de las iteraciones');
xlabel('Iteración');
ylabel('log(likelihood)');


%% Test
test = [a_test;o_test;u_test];
test_real_label = [ones(length(a_test),1);2*ones(length(o_test),1);3*ones(length(u_test),1)];

for k=1:length(test)
		disc(k,1) = discriminante(test(k,:), ma, cova, proba);
		disc(k,2) = discriminante(test(k,:), mo, covo, probo);
		disc(k,3) = discriminante(test(k,:), mu, covu, probu);
		[a,b] = max(disc(k,:));
		test_label(k) = b;
end

% Gráfico
figure()
hold on

% Medias
plot(ma(1),ma(2), 'rx','MarkerSize',10);
plot(mo(1),mo(2),'x','color',myGreen,'MarkerSize',10);
plot(mu(1),mu(2), 'bx','MarkerSize',10);

% Puntos encontrados
plot(test(:,1).*(test_label==1)', test(:,2).*(test_label==1)', 'ro');
plot(test(:,1).*(test_label==2)', test(:,2).*(test_label==2)', 'o', 'color', myGreen);
plot(test(:,1).*(test_label==3)', test(:,2).*(test_label==3)', 'bo');

% Puntos reales
plot(test(:,1).*(test_real_label==1), test(:,2).*(test_real_label==1), 'r.');
plot(test(:,1).*(test_real_label==2), test(:,2).*(test_real_label==2), '.', 'color', myGreen);
plot(test(:,1).*(test_real_label==3), test(:,2).*(test_real_label==3), 'b.');

title('Test')


% plot puntos test, reales versus encontrados.
% Title -> error (en porciento) comparando test_labelreal con test_label
