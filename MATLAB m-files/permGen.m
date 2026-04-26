function [newDogData, perm, fix1, fix2, fix3, fix4, fix5]  = permGen(racePick, raceRows, rowsPerDog, loop, p)


%this gets just the dog data
dogData = racePick((raceRows+1):end);

newDogData = [];
for j = 1:6
   perm = p(loop,:);
   dStartRow = 1+((p(loop,j)-1)*(rowsPerDog));
   dEndRow = dStartRow + (rowsPerDog - 1);
   newDogData = [newDogData;dogData(dStartRow:dEndRow)];
end

[fix1 fix2 fix3 fix4 fix5] = fixedWeightNodes(newDogData, rowsPerDog);

%code here to add on the race data and export as single column
justRaceData = racePick(1:raceRows);
newDogData = [justRaceData;newDogData];






