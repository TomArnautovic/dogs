function dogMain

ZOOM ON;
%ZOOM OUT;



MaxError = 0;
hold off;
%clear axis
cla;

%training/predicting
HH = findobj('tag','buttongroup','string','TRAINING');
Training = get(HH,'Value');

%gets the race lengths
HH = findobj('tag','editRaceLen');
raceLen = str2num(get(HH,'String'));

%number of races exported
HH = findobj('tag','editRace');
Races = str2num(get(HH,'String'));

if Training == 1
Inputz = Converter(raceLen, Races);
end

%gets the number of rows of just race data exluding dog data
HH = findobj('tag', 'editRaceRows');
raceRows = str2num(get(HH, 'String'));

%calculate the rows per dog
rowsPerDog = (raceLen - raceRows)/6;

%get wav sounds
finTrainWav = wavread('training.wav');
finPrediWav = wavread('predicting.wav');

%for plotting average errors
TError = 0;

%Initialize oldError used of Alpha calculation
oldError = 0;

%variable used in plotting
firstPlot = 0;

%factor for weight initilisation
HH = findobj('Tag','editWeight');
weightFactor = str2num(get(HH,'String'));

%get the name of the variable save
HH = findobj('Tag', 'popSave');
netData = num2str(get(HH, 'Value'));

%number of perms. for each race
HH = findobj('tag','editPerm');
Perms = str2num(get(HH,'String'));

%get epochs
HH = findobj('tag','editEpoch');
Epochs = str2num(get(HH,'String'));

%number of races to be predicted
HH = findobj('tag','editRacePred');
noRaces = str2num(get(HH,'String'));

%get alpha
HH = findobj('tag','editAlpha');
Alpha = str2num(get(HH,'String'));

%get Nodes1
HH = findobj('tag','editNeuron1');
Nodes1 = str2num(get(HH,'String'));

%get Nodes2
HH = findobj('tag','editNeuron2');
Nodes2 = str2num(get(HH,'String'));

%old/new weights
HH = findobj('tag','buttongroup2', 'string', 'NEW');
ResetWeights = get(HH, 'Value');

%output node
Nodes3 = 1;
tic;
%training else predicting
if Training == 1
  xlabel ('EPOCHS');
  
%reset weights
	if ResetWeights == 1
     
      %setup Weights
		Weights1 = weightFactor * rand(Nodes1, raceLen);
		Weights2 = weightFactor * rand(Nodes2, Nodes1); 
		Weights3 = weightFactor * rand(1, Nodes2);
            
      %setup bias	
      bias1 = ones(Nodes1,1);
      bias2 = ones(Nodes2,1);
  		bias3 = ones(Nodes3,1);
   else
      
   	load (netData, 'Weights1', 'Weights2', 'Weights3', 'bias1', 'bias2', 'bias3');
	end  
   
   h = waitbar(0,'Please wait...');
   
   for i = 1:Epochs
   
   	waitbar(i/Epochs)   
   	for j = 1:Races
         for k = 1:Perms
            racePick = str2num(Inputz(:,j));
   			[Inputs Target]  = randJumbler(racePick, raceRows, rowsPerDog);
            
           %Feed Forward
				Sum1 = Weights1 * Inputs + bias1;
            Output1 = transf(Sum1);
            
				Sum2 = Weights2 * Output1 + bias2;
            Output2 = transf2(Sum2);
				Sum3 = Weights3 * Output2 + bias3;
            Output3 = purelin(Sum3);
   
            %backprop
           	Delta1 = -2 * dpurelin(1, Output3) * (Target - Output3);
				Delta2 = dtransf2(diag(Output2)) * Weights3' * Delta1;
				Delta3 = dtransf(diag(Output1)) * Weights2' * Delta2;
				
            %update all weight;
            Weights3 = Weights3 - (Alpha * Delta1 * Output2');
            Weights2 = Weights2 - (Alpha * Delta2 * Output1');
           	Weights1 = Weights1 - (Alpha * Delta3 * Inputs');

				bias3 = bias3 - (Alpha * Delta1);
				bias2 = bias2 - (Alpha * Delta2);
				bias1 = bias1 - (Alpha * Delta3);
            
            Error = Target - Output3;

				Alpha = newAlpha(Alpha, Error, oldError, Epochs);
            oldError = Error;
            TError = (TError + abs(Error));
            
         end %End perm Loop
         
         TError = TError/Perms;
  
end %End Race Loop
      
      figure(1);
    
      if TError > MaxError
         MaxError = TError;
      end
      
      hold on;
      set(gca,'XTickMode', 'auto');
      axis([0 Epochs 0 MaxError]);
   
     
     if firstPlot ~= 0
         line([firstX,i],[firstY,TError])
         firstX = i;
         firstY = TError;
      else
         firstPlot = 1;
         firstX = i;
         firstY = TError;
      end 
      
     save (netData, 'Weights1', 'Weights2', 'Weights3', 'bias1', 'bias2', 'bias3')
	
   end %End epoch loop
 
   close(h);
   
   %play wave
   sound(finTrainWav,45000)
   
   %setting radio button to old weights
	HH = findobj('tag', 'buttongroup2','String','OLD');
	set(HH, 'Value',1);
	HH = findobj('tag', 'buttongroup2','String','NEW');
	set(HH, 'Value',0);
else
   
   
   
   
   
   %-----------------------------------------------------------------
   %-----------------------------------------------------------------
   %-----------------------------------------------------------------
   
   %PREDICTING
   load (netData, 'Weights1', 'Weights2', 'Weights3', 'bias1', 'bias2', 'bias3')
   results = [];
   results2 = [];
   results3 = [];
   
   allPerms = 720;
   n2 = 1; %used to split up multiple races
   x = 1; %used for waitbar
   
   p = [];
for i = 1:6
   for j = 1:6
      if i ~= j
         for k = 1:6
         	if k ~= i & k ~=j
               for l = 1:6
                  if l ~= k & l ~= i & l ~= j
                     for m = 1:6
                        if m ~= l & m ~= j & m ~= k & m ~= i                  
                           for n = 1:6
                              if n ~= m & n ~= l & n ~= k & n ~= i & n ~= j
                                 p = [p;i j k l m n];
                              end
               				end
            				end
         				end
      				end
  			 		end
  		 		end
   		end
		end
   end
end
   h = waitbar(0,'Please wait...');
   

   for y2 = 1:noRaces
     	allOuts = [];
      figure(1);
      grid on;
      
      axis([1 720 0 1]);
      HH = gca;
      set(HH, 'XTick',[ 0 120 240 360 480 600 720]);
      hold on;
    	axis manual;

      xlabel ('PERM NUMBER');
   
      
      Inputx = Converter2(raceLen, noRaces); 
      racePick = str2num(Inputx(:,y2));          
      
      bestPerm = [1 2 3 4 5 6];
   	lowOut = 100000;
   
   for j2 = 1:allPerms
        
      waitbar(x/(allPerms * noRaces))
      x = x + 1;
      [Inputy perm] = permGen(racePick, raceRows, rowsPerDog, j2, p);
    

      %Feed Forward
            Sum1 = Weights1 * Inputy + bias1;
            Output1 = transf(Sum1);
            Sum2 = Weights2 * Output1 + bias2;
            Output2 = transf2(Sum2);
				Sum3 = Weights3 * Output2 + bias3;
            Output3 = purelin(Sum3);
            
            allOuts = [allOuts;Output3];
             		
          %shows output for each perm     
          %HH = findobj('Tag','textOutput');
          %set(HH,'String',Output3);
           
      if Output3 < lowOut
         bestPerm = perm;
         lowOut = Output3;
         bestPerm = num2str(bestPerm');
         
         HH = findobj('Tag','textResult');
         set(HH,'String',bestPerm');
         
         HH = findobj('Tag','textLowOutput');
         set(HH,'String',lowOut);
      end
      
   end
   StdDev = std(allOuts);
   AvMean = mean(allOuts);
   
   %stores result for each race
   results(y2,:) = [str2num(bestPerm')];
   
   %stores stdDev for each race
   results2(y2,:) = [StdDev];
   
   %stores AvMean for each race
   results3(y2,:) = [lowOut];
   
   figure(1);
   permNums = [1:720]';
   plot(permNums, allOuts);
   %bestPerm
   %lowOut

     
   HH = findobj('Tag','textStdDev');
   set(HH, 'String', StdDev);
   HH = findobj('Tag','textMean');
   set(HH, 'String',AvMean);

   
   
   HH = findobj('Tag','checkPause');
   predWait = get(HH,'Value');
   
   if predWait == 1
      pause
   end
   cla;
end
close(h);
sound(finPrediWav,45000)
results
results2
results3
end
