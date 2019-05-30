%ej.m
clear all
close all

% Frames de las vocales y se obtienen los 3 formantes. Usamos sólo los 2 primeros para poder verlo mejor.
A = dlmread('a.txt', '\t', 0, 0);
O = dlmread('o.txt', '\t', 0, 0);
U = dlmread('u.txt', '\t', 0, 0);

% Saco el 3er formante de las matrices.
A = A(:,1:2); O = O(:, 1:2); U = U(:, 1:2);

% Separo el set de entrenamiento y testeo
A_train = A(1:35,:);
O_train = O(1:35,:);
U_train = U(1:35,:);

A_test = A(36:end,:);
O_test = O(36:end,:);
U_test = U(36:end,:);

% Hago el 'scatter' de las muestras.
figure
hold on
plot(A_train(:,1), A_train(:,2), 'bo')
plot(O_train(:,1), O_train(:,2), 'ro')
plot(U_train(:,1), U_train(:,2), 'o','color',[0 .5 0])
grid minor


% Calculo los parámetros
% Media
muA = mean(A_train);
muO = mean(O_train);
muU = mean(U_train);
plot(muA(1), muA(2), 'bo', 'MarkerSize', 17)
plot(muO(1), muO(2), 'ro', 'MarkerSize', 17)
plot(muU(1), muU(2), 'o', 'color', [0 .5 0], 'MarkerSize', 17)


% Sigma
%	sigmaA = mean((A_train-muA)*(A_train-muA)');
%	sigmaO = mean((O_train-muO)*(O_train-muO)');
%	sigmaU = mean((U_train-muU)*(U_train-muU)');
%	sigma = (sigmaA + sigmaO + sigmaU)/3;

sigmaA = estim_sigma(A_train, muA);
sigmaO = estim_sigma(O_train, muO);
sigmaU = estim_sigma(U_train, muU);
Sigma = (sigmaA + sigmaO + sigmaU)/3;

% Dibujo las elipses
theta = linspace(0, 2*pi, 100);
rot = [sin(theta); cos(theta)];
elipseA = (chol(sigmaA)' * rot)' + muA;	% Ojo que no broadcastea en matlab (matriz+vector)
elipseO = (chol(sigmaO)' * rot)' + muO;
elipseU = (chol(sigmaU)' * rot)' + muU;

plot(elipseA(:,1), elipseA(:,2),'b')
plot(elipseO(:,1), elipseO(:,2),'r')
plot(elipseU(:,1), elipseU(:,2),'color', [0 .5 0])

% Clasifico según la ecuación de la recta que vimos en clase pero antes de que sea la recta
% Agarramos las igualadas (no a cero) y descarto el término cuadrático (xT sigma^-1 x), de
% modo que tenemos una ecuación de discriminante. Se saca porque se quiere comparar y ése término
% queda en todos las ecuaciones. Se decide que es la clase donde el discriminante es mayor.
NA = length(A_train); NO = length(O_train); NU = length(U_train); N = NA + NO + NU;

%slopeA = (inv(sigma)*muA)';	constA = -1/2 * muA' * inv(sigma) * muA;	probA = log(NA/N);
%slopeU = (inv(sigma)*muU)';	constU = -1/2 * muU' * inv(sigma) * muU;	probU = log(NU/N);
%slopeO = (inv(sigma)*muO)';	constO = -1/2 * muO' * inv(sigma) * muO;	probO = log(NO/N);

%% Clasifico los test de A
a_bien = zeros(1,2);	a_mal = zeros(1,2);
for i = 1:length(A_test)
	% Podría almacenarlo (disc(i)) para ver quién ganó
	discrA = discriminante(A_test(i,:), muA, Sigma, log(NA/N));
	discrU = discriminante(A_test(i,:), muU, Sigma, log(NU/N));
	discrO = discriminante(A_test(i,:), muO, Sigma, log(NO/N));

	% Chequeo si discA > los demas
	if((discrA > discrU) && (discrA > discrO))
		% Clasificó bien
		a_bien = [a_bien; A_test(i,:)];
	else
		% Clasificó mal
		a_mal = [a_mal; A_test(i,:)];
	end
end
%a_bien = a_bien(2:end,:);	a_mal = a_mal(2:end,:);
% No lo borro porque ayudan a graficar (ejes)

figure; hold on
plot(a_bien(:,1), a_bien(:,2), 'bx')
plot(a_mal(:,1), a_mal(:,2), 'ko')

%% Clasifico los test de U
u_bien = zeros(1,2);	u_mal = zeros(1,2);
for i = 1:length(U_test)
	% Podría almacenarlo (disc(i)) para ver quién ganó
	discrA = discriminante(U_test(i,:), muA, Sigma, log(NA/N));
	discrU = discriminante(U_test(i,:), muU, Sigma, log(NU/N));
	discrO = discriminante(U_test(i,:), muO, Sigma, log(NO/N));

	% Chequeo si discA > los demas
	if((discrA < discrU) && (discrU > discrO))
		% Clasificó bien
		u_bien = [u_bien; U_test(i,:)];
	else
		% Clasificó mal
		u_mal = [u_mal; U_test(i,:)];
	end
end
%u_bien = u_bien(2:end,:);	u_mal = u_mal(2:end,:);

figure; hold on
plot(u_bien(:,1), u_bien(:,2), 'rx')
plot(u_mal(:,1), u_mal(:,2), 'ko')

%% Clasifico los test de O
o_bien = zeros(1,2);	o_mal = zeros(1,2);
for i = 1:length(O_test)
	% Podría almacenarlo (disc(i)) para ver quién ganó
	discrA = discriminante(O_test(i,:), muA, Sigma, log(NA/N));
	discrU = discriminante(O_test(i,:), muU, Sigma, log(NU/N));
	discrO = discriminante(O_test(i,:), muO, Sigma, log(NO/N));

	% Chequeo si discA > los demas
	if((discrA < discrO) && (discrU < discrO))
		% Clasificó bien
		o_bien = [o_bien; O_test(i,:)];
	else
		% Clasificó mal
		o_mal = [o_mal; O_test(i,:)];
	end
end
%o_bien = o_bien(2:end,:);	o_mal = o_mal(2:end,:);

figure; hold on
plot(o_bien(:,1), o_bien(:,2), 'x', 'color', [0 .5 0])
plot(o_mal(:,1), o_mal(:,2), 'ko')
