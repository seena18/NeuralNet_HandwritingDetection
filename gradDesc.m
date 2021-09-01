function [Theta1,Theta2] = gradDesc(lr,Theta1,Theta2,grad1,grad2)
Theta1=Theta1-lr*grad1;
Theta2=Theta2-lr*grad2;