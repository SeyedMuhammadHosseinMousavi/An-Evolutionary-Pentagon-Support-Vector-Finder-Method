clc;
clear;
close all;
%% Fuzzy C Means Clustering
% Loading Data
load BloodTransfusion.mat;
BloodT=NewTextDocument;
BloodTDat=BloodT(:,1:end-1);
BloodTLbl=BloodT(:,end);
X = BloodTDat;
OriginData=X;
% U is degree of membership
FCMclus=350;
[centers,U] = fcm(X,FCMclus);
% Plot original and after FCM
% Matrix of subaxes containing scatter plots of the columns of X against
% the columns of Y
% plotmatrix(meas,'bh');
% title('Origignal');
% figure;
% plotmatrix(centers,'k.');
% title('After FCM');
FCMData=centers;
%% ABC + FCM Clustering
Method = 'CS';	% DB or CS
X = FCMData;
% K is number of clusters
k = 150;
CostFunction=@(s) ClusteringCost(s, X, Method);     % Cost Function
VarSize=[k size(X,2)+1];  % Decision Variables Matrix Size
nVar=prod(VarSize);     % Number of Decision Variables
VarMin= repmat([min(X) 0],k,1);      % Lower Bound of Variables
VarMax= repmat([max(X) 1],k,1);      % Upper Bound of Variables
% ABC Settings
MaxIt=50;              % Maximum Number of Iterations
nPop=6;               % Population Size (Colony Size)
nOnlooker=nPop;         % Number of Onlooker Bees
L=round(0.5*nVar*nPop); % Abandonment Limit Parameter
a=1;                    % Acceleration Coefficient Upper Bound
% Initialization
% Empty Bee Structure
empty_bee.Position=[];
empty_bee.Cost=[];
empty_bee.Out=[];
% Initialize Population Array
pop=repmat(empty_bee,nPop,1);
% Initialize Best Solution Ever Found
BestSol.Cost=inf;
% Create Initial Population
for i=1:nPop
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    [pop(i).Cost, pop(i).Out]=CostFunction(pop(i).Position);
    
    if pop(i).Cost<=BestSol.Cost
        BestSol=pop(i);
    end
end
% Abandonment Counter
C=zeros(nPop,1);
% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);
% ABC Main Loop
for it=1:MaxIt
    % Recruited Bees
    for i=1:nPop        
        % Choose k randomly, not equal to i
        K=[1:i-1 i+1:nPop];
        k=K(randi([1 numel(K)]));        
        % Define Acceleration Coeff.
        phi=unifrnd(-a,+a,VarSize);        
        % New Bee Position
        newbee.Position=pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
        newbee.Position=max(newbee.Position, VarMin);
        newbee.Position=min(newbee.Position, VarMax);        
        % Evaluation
        [newbee.Cost, newbee.Out]=CostFunction(newbee.Position);       
        % Comparision
        if newbee.Cost<=pop(i).Cost
            pop(i)=newbee;
        else
            C(i)=C(i)+1;
        end
        
    end    
    % Calculate Fitness Values and Selection Probabilities
    F=zeros(nPop,1);
    for i=1:nPop
        if pop(i).Cost>=0
            F(i)=1/(1+pop(i).Cost);
        else
            F(i)=1+abs(pop(i).Cost);
        end
    end
    P=F/sum(F);    
    % Onlooker Bees
    for m=1:nOnlooker        
        % Select Source Site
        i=RouletteWheelSelection(P);       
        % Choose k randomly, not equal to i
        K=[1:i-1 i+1:nPop];
        k=K(randi([1 numel(K)]));
        % Define Acceleration Coeff.
        phi=unifrnd(-a,+a,VarSize);        
        % New Bee Position
        newbee.Position=pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
        newbee.Position=max(newbee.Position, VarMin);
        newbee.Position=min(newbee.Position, VarMax);        
        % Evaluation
        [newbee.Cost, newbee.Out]=CostFunction(newbee.Position);        
        % Comparision
        if newbee.Cost<=pop(i).Cost
            pop(i)=newbee;
        else
            C(i)=C(i)+1;
        end       
    end    
    % Scout Bees
    for i=1:nPop
        if C(i)>=L
            pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
            [pop(i).Cost, pop(i).Out]=CostFunction(pop(i).Position);
            C(i)=0;
        end
    end    
    % Update Best Solution Ever Found
    for i=1:nPop
        if pop(i).Cost<=BestSol.Cost
            BestSol=pop(i);
        end
    end   
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;    
    % Display Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);    
    % Plot Solution
    figure(1);
    PlotSolution(X, BestSol);
    pause(0.01);
end   
% Results
figure;
plot(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
% Plot original and after FCM
% Matrix of subaxes containing scatter plots of the columns of X against
% the columns of Y
figure;
plotmatrix(BloodTDat,'bh');
title('Origignal');
figure;
plotmatrix(centers,'k.');
title('After FCM');
figure;
ABCClusRes=BestSol.Position(:,1:4);
plotmatrix(ABCClusRes,'r*');
title('After ABC + FCM');
%% K-Means and K-NN for labels and outliers removal (first step)
rng(1); % For reproducibility
[idx,C] = kmeans(ABCClusRes,2);
% K-NN labels Predict - k-nearest neighbor
% Create a classifier for ? nearest neighbors
mdl = fitcknn(ABCClusRes,idx,'NumNeighbors',2,'Standardize',1);
% Predict labels 
[label,score,cost] = predict(mdl,ABCClusRes)
% Removing outliers (first step) based on moving window.
% Outliers are defined as elements more than three local standard deviations
% from the local mean over a window length specified by window
OutlierRemFirst = rmoutliers([ABCClusRes label],'movmedian',2);
NewData=OutlierRemFirst(:,1:end-1);
NewLabel=OutlierRemFirst(:,end);
figure;
plotmatrix(NewData,'g+');
title('After K-NN Window Labeling and Outlier Removal');
% Deviding classes and labels
sizedata=size(OutlierRemFirst);
class1(1:sizedata(1,2))=0;
class2(1:sizedata(1,2))=0;
for i=1:sizedata(1,1)
    if NewLabel(i)== 1
        class1(i,1:end)= OutlierRemFirst(i,:);
    elseif NewLabel(i)== 2
        class2(i,1:end)= OutlierRemFirst(i,:);
    end
end
% Removing zero rows
indc1 = find(sum(class1,2)==0) ;
indc2 = find(sum(class2,2)==0) ;
class1(indc1,:) = [] ;
class2(indc2,:) = [] ;
% Pentagon Support Vector Finder Step
psvdiv=floor(sizedata(1:1)/5);
j=1;
for i=1:psvdiv
psvdat{i}=OutlierRemFirst(j:j+4,1:end-1);
j=j+5;
end;
% area
for i=1:psvdiv
    penarea(i)=polyarea(psvdat{i}(:,1),psvdat{i}(:,2))+polyarea(psvdat{i}(:,3),psvdat{i}(:,4));
end;
% angle
for i=1:psvdiv
    penangle{i}=regionprops(psvdat{i},'MaxFeretProperties');
    atemp(i)=penangle{i}.MaxFeretAngle;
    ang(i)=abs(atemp(i));
end;
%
j=1;
for i=1:psvdiv
forfin{i}=OutlierRemFirst(j:j+4,:);
j=j+5;
end;
%
threshold=6.804978501753682e+04;
for i=1:psvdiv
    if penarea(i)<threshold && ang(i)>=0
        newpsvdat{i}=forfin{i};
    else
        newpsvdat{i}=0;
    end
end
%
FinalReady=newpsvdat;
% Removing Zeros to Empty
for i = 1:psvdiv
FinalReady{i}(FinalReady{i}==0) = [];
end
% Removingg Empty
FinalReady = FinalReady(~cellfun(@isempty, FinalReady));
% Final Matrix with Label
FinalReady = cell2mat(FinalReady');
% Plot
figure;
plotmatrix(FinalReady(:,1:end-1),'kd');
title('After PSV');
% Classification
% KNN 
lblknn=FinalReady(:,end);
dataknn=FinalReady(:,1:end-1);
Mdl = fitcknn(dataknn,lblknn,'NumNeighbors',5,'Standardize',1)
rng(1); % For reproducibility
knndat = crossval(Mdl);
classError = kfoldLoss(knndat)
% SVM
svmclass = fitcecoc(dataknn,lblknn);
svmerror = resubLoss(svmclass);
CVMdl = crossval(svmclass);
genError = kfoldLoss(CVMdl);
% Shallow Neural Network
network=FinalReady(:,1:end-1);
netlbl=FinalReady(:,end);
sizenet=size(network);
sizenet=sizenet(1,1);
for i=1 : sizenet
            if netlbl(i) == 1
               netlbl2(i,1)=1;
        elseif netlbl(i) == 2
               netlbl2(i,2)=1;
        end
end
% Changing data shape from rows to columns
network=network'; 
% Changing data shape from rows to columns
netlbl2=netlbl2'; 
% Defining input and target variables
inputs = network;
targets = netlbl2;
% Create a Pattern Recognition Network
hiddenLayerSize = 100;
net = patternnet(hiddenLayerSize);
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% Train the Network
% Polak-Ribiére Conjugate Gradient
net = feedforwardnet(20, 'traincgp');
%
[net,tr] = train(net,inputs,targets);
% Test the Network
outputs = net(inputs);
%
errors = gsubtract(targets,outputs);
%
performance = perform(net,targets,outputs)
% Polak-Ribiére Conjugate Gradient
figure, plottrainstate(tr)
% Plot Confusion Matrixes
figure, plotconfusion(targets,outputs);
title('Polak-Ribiére Conjugate Gradient');
% Res
disp(['K-NN Classification Accuracy :   ' num2str(100-classError) ]);
disp(['SVM Classification Accuracy :   ' num2str(100-genError) ]);
disp(['NN Classification Accuracy :   ' num2str(100-performance) ]);






