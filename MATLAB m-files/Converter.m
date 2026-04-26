function inputString = Converter(raceLen, Races)

HH = findobj('tag','editData');
exportData = get(HH,'String');
fid = fopen(exportData);

inputString = fscanf(fid,'%s',[raceLen Races]);
   
fclose(fid);

