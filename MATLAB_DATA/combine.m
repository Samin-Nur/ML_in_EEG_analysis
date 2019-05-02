load('Feature_set.mat')
load('Labels.mat')

X=Feature_set;
X=X.';
TrainX=X(1:900,:);
TestX=X(901:1200,:);

Y=Output;

for i=1:1200
    if Y(i)==0
        Y(i)=1;
    end
end
Y=Y.';
TrainY=Y(1:900);
TestY=Y(901:1200);


save('TrainX.mat','TrainX')
save('TestX.mat','TestX')
save('TrainY.mat','TrainY')
save('TestY.mat','TestY')
