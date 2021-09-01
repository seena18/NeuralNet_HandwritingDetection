function [J, Theta1_grad, Theta2_grad] = nnCostFunction(Theta1,Theta2, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                                 
%Calculate cost function and implement back propogation

                 
m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
accumulator1 = zeros(size(Theta1));
accumulator2 = zeros(size(Theta2));

X = [ones(m, 1) X];

if lambda == 0
  for i = 1:m

    a1=X(i:i,1:end);
    z2=Theta1*a1';
    a2=sigmoid(z2);
    a2=a2';
    a2 = [ones(size(a2,1),1) a2];
   
    a2=a2';
    
    z3=Theta2*a2;
    a3=sigmoid(z3);
    
    hx=a3;
    p=zeros(1,num_labels);
    p(y(i))=1;
    p=p';
    J=J+((-p'*log(hx)-(1-p')*log(1-hx)));
    del3=a3-p;
    z2=z2';
    ztwo=[ones(size(z2,1),1) z2];
    ztwo=ztwo';

    del2=((Theta2)'*del3).*sigmoidGradient(ztwo);
    

    del2=del2(2:end);
    accumulator2=accumulator2+del3*a2';
    accumulator1=accumulator1+del2*a1;

    
  end
  J=J/m;
  Theta1_grad=accumulator1./m;
  Theta2_grad=accumulator2./m;
  
else
    for i = 1:m
    a1=X(i:i,1:end);
    z2=Theta1*a1';
    a2=sigmoid(z2);
    a2=a2';
    a2 = [ones(size(a2,1),1) a2];
    a2=a2';
    z3=Theta2*a2;
    a3=sigmoid(z3);
    
    hx=a3;
    p=zeros(1,num_labels);
    p(y(i))=1;
    p=p';
    J=J+((-p'*log(hx)-(1-p')*log(1-hx)));
    del3=a3-p;
    z2=z2';
    ztwo=[ones(size(z2,1),1) z2];
    ztwo=ztwo';

    del2=((Theta2)'*del3).*sigmoidGradient(ztwo);
    

    del2=del2(2:end);
    accumulator2=accumulator2+del3*a2';
    accumulator1=accumulator1+del2*a1;
    
    
  end
      temptheta1=Theta1(1:end,2:end);
      temptheta2=Theta2(1:end,2:end);
      reg = (lambda/(2*m))*((sum(temptheta1(:).^2))+(sum(temptheta2(:).^2)));
      J=(J/m)+reg;
     
      Theta1_grad=accumulator1./m;
      Theta2_grad=accumulator2./m;
      Theta1_grad(1:end,2:end)=Theta1_grad(1:end,2:end)+((lambda *Theta1(1:end,2:end))/m);
      Theta2_grad(1:end,2:end)=Theta2_grad(1:end,2:end)+((lambda *Theta2(1:end,2:end))/m);
endif