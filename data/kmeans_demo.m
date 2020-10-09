clear all; close all; clc;
warning off all;

K_value = 3;    % the number of clusters

rng default;    % For reproducibility (set the random number generator seed)

% The sample data
% X = [10  2 13 20 17  8  4 19 11 15 7  1;     % first row: x coordinats
%      1 19 12 15  2 13 14  3 18  7 8 20]';   % second row: y coordinates

disp('We start with the following set of 2D points:');
X = randi(30,12,2)

% Randomly initialize cluster centers
disp('Starting cluster centroid locations:');
C = randi(max(X(:)),K_value,2)

% Plot the sample data
figure;
plot(X(:,1),X(:,2),'.');
title 'A set of 2D Points';

for maxiter = 1:3
    disp(['Iteration# ' num2str(maxiter)]);
    disp('The following are the Cluster Assignments and Centroids:'); 
    
    [idx,C] = kmeans(X,[],'Start',C,'MaxIter',maxiter)

    % Plot the clusters and the cluster centroids
    figure;
    hold on

    for loop_idx = 1:K_value
        leg = strcat('Cluster ',int2str(loop_idx));
        plot(X(idx==loop_idx,1),X(idx==loop_idx,2),'.','MarkerSize',12,'DisplayName',leg);
    end
    plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3,'DisplayName','Centroids');
    legend('Location','southwest');
    legend show;

    title(strcat('Cluster Assignments and Centroids after Iteration# ', num2str(maxiter)))
    hold off
    
    disp('Press any key to continue...');
    pause;
end

warning on all;
