function transfer = dtransf2(x)

%get transfer function
HH = findobj('tag','popNeuron2');
Ftype = get(HH, 'Value');

switch Ftype
     	case {1}
        	transfer = dhardlim(x);
      case {2}
         transfer = dhardlms(x);
      case {3}
         transfer = dpurelin(x); 
      case {4}
         transfer = dsatlin(x); 
      case {5}
         transfer = dsatlins(x);
      case {6}
         transfer = dlogsig(1,x);
      case {7}
         transfer = dtansig(1,x);
      case {8}
   	   transfer = dposlin(x);
end