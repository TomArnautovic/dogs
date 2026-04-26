function transfer = transf2(x)

%get transfer function
HH = findobj('tag','popNeuron2');
Ftype = get(HH, 'Value');

switch Ftype
     	case {1}
        	transfer = hardlim(x);
      case {2}
         transfer = hardlims(x); 
      case {3}
         transfer = purelin(x); 
      case {4}
         transfer = satlin(x);
      case {5}
         transfer = satlins(x);
      case {6}
         transfer = logsig(x);
      case {7}
         transfer = tansig(x);
      case {8}
   	   transfer = poslin(x); 
end