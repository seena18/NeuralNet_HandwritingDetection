clear ; close all; clc
input_layer_size  = 400;  
hidden_layer_size = 25;  
num_labels = 10;
lambda = 1;
lr =.1;

load('ex4data1.mat');
m = size(X, 1);




userInput = input('Load weights(Y/n)', 's');
if userInput=='y'
  fprintf('loading weights...');
  load('theta1.mat');
  load('theta2.mat');
else
  fprintf('Generating random weights to train model...\n');
  Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
  Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
endif


for i = 1:10000
[J, grad1, grad2] = nnCostFunction(Theta1,Theta2, input_layer_size, hidden_layer_size,num_labels,X,y,lambda);
[Theta1,Theta2] = gradDesc(lr,Theta1,Theta2,grad1,grad2);
fprintf('Iteration:%d \n %f \n',i,J);
if mod(i,10)==0 
  pred = predict(Theta1, Theta2, X);
                 
  fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
  %save('theta1.mat','Theta1');
  %save('theta2.mat','Theta2');
endif
end

