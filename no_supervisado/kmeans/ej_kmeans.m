%% K-means

clear all
close all
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

%% iteraciones:
% Medias de las clases
ma = ma_inic;
mo = mo_inic;
mu = mu_inic;

% Label
real_label = [ones(length(a_train),1);2*ones(length(o_train),1);3*ones(length(u_train),1)];

distorsion(1) = inf;		% La distorsión inicial pongo inf para que arranque el algoritmo
distorsion(2) = 10000;	% Siendo la segunda MUY grande
n = 2;									% Se inicializa con 2 porque ya cargué 2 distorsiones
limdist = 0.01;

%% Preparo el gráfico

figure(1); hold on;
% Medias con X
plot(ma(1),ma(2), 'bx','MarkerSize',10);
plot(mo(1),mo(2), 'rx','MarkerSize',10);
plot(mu(1),mu(2),'x','color',myGreen,'MarkerSize',10);
% Puntos bien clasificados
plot(train(:,1).*(real_label==1), train(:,2).*(real_label==1), 'b.', 'MarkerSize',10);
plot(train(:,1).*(real_label==2), train(:,2).*(real_label==2), 'r.', 'MarkerSize',10);
plot(train(:,1).*(real_label==3), train(:,2).*(real_label==3), '.', 'color', myGreen, 'MarkerSize',10);
titles = ['Iter0'; 'Iter1'; 'Iter2'; 'Iter3'; 'Iter4'; 'Iter5'; 'Iter6'; 'Iter7'];
title(titles(1,:));

bool_plot = 1;

while abs((distorsion(n-1) - distorsion(n))) > limdist
	for k = 1:length(train)
		dist = [train(k,:)-ma; train(k,:)-mo; train(k,:)-mu]; % Calculo la distancia del punto a cada media
		dist = vecnorm(dist');	% Uso vecnorm que me hace la norma, suponiendo que es un arreglo de vectores
		[val, label(k)] = min(dist);	% El que tenga distancia mínima es al que le asigno esa clase
	end

	% Recalculo la media
	ma = sum(train.*(label==1)')/(sum(label==1));
	mo = sum(train.*(label==2)')/(sum(label==2));
	mu = sum(train.*(label==3)')/(sum(label==3));

	% Recalculo la distorsión
	distorsion_a = sum(vecnorm(((train-ma).*(label==1)')'))/(sum(label==1));
	distorsion_o = sum(vecnorm(((train-mo).*(label==2)')'))/(sum(label==2));
	distorsion_u = sum(vecnorm(((train-mu).*(label==3)')'))/(sum(label==3));
	distorsion(n+1) = distorsion_a + distorsion_o + distorsion_u;

	if(bool_plot)
		% Grafico
		figure(n)
		hold on

		% Medias
		plot(ma(1),ma(2), 'bx','MarkerSize',10);
		plot(mo(1),mo(2), 'rx','MarkerSize',10);
		plot(mu(1),mu(2),'x','color',myGreen,'MarkerSize',10);

		% Puntos reales
		plot(train(:,1).*(real_label==1), train(:,2).*(real_label==1), 'b.', 'MarkerSize',10);
		plot(train(:,1).*(real_label==2), train(:,2).*(real_label==2), 'r.', 'MarkerSize',10);
		plot(train(:,1).*(real_label==3), train(:,2).*(real_label==3), '.', 'color', myGreen, 'MarkerSize',10);

		% Puntos clasificados
		plot(train(:,1).*(label==1)', train(:,2).*(label==1)', 'bo', 'MarkerSize',10);
		plot(train(:,1).*(label==2)', train(:,2).*(label==2)', 'ro', 'MarkerSize',10);
		plot(train(:,1).*(label==3)', train(:,2).*(label==3)', 'o', 'color', myGreen, 'MarkerSize',10);

		xlabel('Primer formante [Hz]');
		ylabel('Segundo formante [Hz]');
		title(titles(n,:));
	end

	n = n+1;
end

close all


%% Plots de datos tras entrenamiento
% para las elipses!
cova = estim_sigma(train.*(label==1)', ma, sum(label==1));
covo = estim_sigma(train.*(label==2)', mo, sum(label==2));
covu = estim_sigma(train.*(label==3)', mu, sum(label==3));
theta = linspace(0, 2*pi, 100);
rot = [sin(theta); cos(theta)];

elipsea = (chol(cova)' * rot)' + ma;
elipseo = (chol(covo)' * rot)' + mo;
elipseu = (chol(covu)' * rot)' + mu;

% Gráfico
figure()
hold on

% Medias
plot(ma(1),ma(2), 'bx','MarkerSize',10);
plot(mo(1),mo(2), 'rx','MarkerSize',10);
plot(mu(1),mu(2),'x','color',myGreen,'MarkerSize',10);

% Puntos encontrados
plot(train(:,1).*(label==1)', train(:,2).*(label==1)', 'bo');
plot(train(:,1).*(label==2)', train(:,2).*(label==2)', 'ro');
plot(train(:,1).*(label==3)', train(:,2).*(label==3)', 'o', 'color', myGreen);

% Elipses
plot(elipsea(:,1), elipsea(:,2), 'b')
plot(elipseo(:,1), elipseo(:,2), 'r')
plot(elipseu(:,1), elipseu(:,2), 'color', myGreen)

title('Elipses, ¿está bien?')

%% Plot distorsion
figure
plot(distorsion);
title('Distorsión en función de las iteraciones');
xlabel('Iteración');
ylabel('Distorsión');


%% Test
test = [a_test;o_test;u_test];
test_real_label = [ones(length(a_test),1);2*ones(length(o_test),1);3*ones(length(u_test),1)];
test_real_label = test_real_label';

NA = sum(label==1);
NO = sum(label==2);
NU = sum(label==3);
N = length(label);

% ¿Debería en cada iteración sumarle al NX que corresponda?
for i=1:length(test)
		disc(i,1) = discriminante(test(i,:), ma, cova, log(NA/N));
		disc(i,2) = discriminante(test(i,:), mo, covo, log(NO/N));
		disc(i,3) = discriminante(test(i,:), mu, covu, log(NU/N));
		[a,b] = max(disc(i,:));
		test_label(i) = b;
end

% Gráfico
figure()
hold on

% Medias
plot(ma(1),ma(2), 'bx','MarkerSize',15);
plot(mo(1),mo(2), 'rx','MarkerSize',15);
plot(mu(1),mu(2),'x','color',myGreen,'MarkerSize',15);

% Puntos encontrados
plot(test(:,1).*(test_label==1)', test(:,2).*(test_label==1)', 'bo','MarkerSize',10);
plot(test(:,1).*(test_label==2)', test(:,2).*(test_label==2)', 'ro','MarkerSize',10);
plot(test(:,1).*(test_label==3)', test(:,2).*(test_label==3)', 'o', 'color', myGreen,'MarkerSize',10);

% Puntos reales
plot(test(:,1).*(test_real_label==1)', test(:,2).*(test_real_label==1)', 'b.','MarkerSize',10);
plot(test(:,1).*(test_real_label==2)', test(:,2).*(test_real_label==2)', 'r.','MarkerSize',10);
plot(test(:,1).*(test_real_label==3)', test(:,2).*(test_real_label==3)', '.', 'color', myGreen,'MarkerSize',10);



title('Test')


% plot puntos test, reales versus encontrados.
% Title -> error (en porciento) comparando test_labelreal con test_label
