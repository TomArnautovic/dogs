function alpha	= newAlpha(Alpha, Error, oldError, Epochs) 

HH = findobj('Tag','popAlpha');
alphaType = get(HH,'Value');

switch alphaType
     	case {1}
           alpha = Alpha;
           
           
      case {2}
         alpha = Alpha - (0.09999/Epochs);
         
      case {3}
      	if Error ~= 0 & oldError ~= 0
      		alpha = Alpha*abs((oldError/(Error))^2);
      	else   
            oldError = Error;
            alpha = Alpha;
   		end
      
 end
      