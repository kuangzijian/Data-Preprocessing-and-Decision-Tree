%Loading dataset
load ovariancancer;
%dataset characteristics
whos
%generate the exact same results everyday
rng(8000,'twister');
%holdout validation, 56 samples are testing and 160 are training data
holdoutCVP = cvpartition(grp,'holdout',56)
dataTrain = obs(holdoutCVP.training,:);
grpTrain = grp(holdoutCVP.training);

%run learning algorith on original featur space
try
   yhat = classify(obs(test(holdoutCVP),:), dataTrain, grpTrain,'quadratic');
catch ME
   display(ME.message);
end
%-------------------------------------------------------------------------
% FILTEER METHOD

%Detcting feature significance by t-test and compare the p-value score 
dataTrainG1 = dataTrain(grp2idx(grpTrain)==1,:);
dataTrainG2 = dataTrain(grp2idx(grpTrain)==2,:);
[h,p,ci,stat] = ttest2(dataTrainG1,dataTrainG2,'Vartype','unequal');
ecdf(p);
xlabel('P value');
ylabel('CDF value')

% How to choose which feature we will keep
% Sorting but where is the cutoff
% Domain knowledge
% Or decide based on classification error

[~,featureIdxSortbyP] = sort(p,2); % sort the features
testMCE = zeros(1,14);
%number of features
nfs = 5:5:70; 
% the same as: %  function err = classf(xtrain,ytrain,xtest,ytest)
               %  yfit = classify(xtest,xtrain,ytrain,'quadratic');
               %  err = sum(~strcmp(ytest,yfit));
classf = @(xtrain,ytrain,xtest,ytest) ...
             sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'quadratic')));
for i = 1:14
   fs = featureIdxSortbyP(1:nfs(i));
   testMCE(i) = crossval(classf,obs(:,fs),grp,'partition',holdoutCVP)/holdoutCVP.TestSize;
end
 plot(nfs, testMCE,'o');
 xlabel('Number of Features');
 ylabel('MCE');
 legend({'MCE on the test set'},'location','NW');
 title('Simple Filter Feature Selection Method');
 %-------------------------------------------------------------------------
 % WRAPPER METHOD
 
 % Problem with filter method
 corr(dataTrain(:,featureIdxSortbyP(1)),dataTrain(:,featureIdxSortbyP(2)))
 
 tenfoldCVP = cvpartition(grpTrain,'kfold',10)
 % search the 150 beest features from filter method
 fs1 = featureIdxSortbyP(1:150);
 %Forward sequential feature selection
 fsLocal = sequentialfs(classf,dataTrain(:,fs1),grpTrain,'cv',tenfoldCVP);
 fs1(fsLocal)
 
 testMCELocal = crossval(classf,obs(:,fs1(fsLocal)),grp,'partition',holdoutCVP)/holdoutCVP.TestSize