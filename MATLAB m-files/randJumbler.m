function [newDogData, sqrDif, fix1, fix2, fix3, fix4, fix5] = randJumbler(racePick, raceRows, rowsPerDog)

dogOrder = [1 2 3 4 5 6];
power = 2;
sqrDif = 0;

%this gets just the dog data
dogData = racePick((raceRows+1):end);

%Generates a random permutation of the dog order
dogPerm = randperm(6);

%calculate squared mean difference
sqrDif = 0;
for i = 1:6
	sqrDif =  (sqrDif + (dogPerm(i)-dogOrder(i))*(dogPerm(i)-dogOrder(i)));
end

sqrDif = sqrDif/70;

%changes dog order

newDogData = [];

for j = 1:6
   dStartRow = 1+((dogPerm(j)-1)*(rowsPerDog));
   dEndRow = dStartRow + (rowsPerDog - 1);
   newDogData = [newDogData;dogData(dStartRow:dEndRow)];
end
[fix1 fix2 fix3 fix4 fix5] = fixedWeightNodes(newDogData, rowsPerDog);

%code here to add on the race data and export as single column
justRaceData = racePick(1:raceRows);
newDogData = [justRaceData;newDogData];



   




