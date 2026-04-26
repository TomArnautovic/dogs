function inputString = Converter2(raceLen, noRaces)

%looks at gui file location
HH = findobj('tag','editData2');
exportData2 = get(HH,'String');
fid = fopen(exportData2);
inputString = fscanf(fid,'%s',[raceLen noRaces]);
fclose(fid);




   
