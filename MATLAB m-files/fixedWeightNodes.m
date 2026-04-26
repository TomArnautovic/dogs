function [fix1, fix2, fix3, fix4, fix5] = fixedWeightNodes(newDogData,rowsPerDog)

x = newDogData;
y = rowsPerDog;

dog1 = x(1:y);
dog2 = x(y+1:2*y);
dog3 = x(2*y+1:3*y);
dog4 = x(3*y+1:4*y);
dog5 = x(4*y+1:5*y);
dog6 = x(5*y+1:6*y);

fix1 = x & [dog2;dog3;dog4;dog5;dog6;dog1];
fix2 = x & [dog3;dog4;dog5;dog6;dog1;dog2];
fix3 = x & [dog4;dog5;dog6;dog1;dog2;dog3];
fix4 = x & [dog5;dog6;dog1;dog2;dog3;dog4];
fix5 = x & [dog6;dog1;dog2;dog3;dog4;dog5];