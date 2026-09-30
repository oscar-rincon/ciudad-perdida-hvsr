function [ZZZ,Fs,time]=load_minishark(fichier)
%[ZZZ,Fs]=load_city(fichier)
%Opens a Cityshark II file and gives a matrix ZZZ with nt rows (number of samples), ncomp columns (number of components) 
%and the sampling frequency Fs. ZZZ is in m/s, gain is corrected, transfert
%function is for Le3D5s.

%Controls the number of headerlines that varies depending on the Readcity version.
%fichier=[char(path),fichier];
  [ti tim time]=textread(fichier,'%s %s %s',1,'headerlines',3);
  [samp ra ra2 Fs]=textread(fichier,'%s %s %s %f',1,'headerlines',8);
  [co factor fact]=textread(fichier,'%s %s %f',1,'headerlines',14);
  [ga gain]=textread(fichier,'%s %f',1,'headerlines',15);

maximum=[];
ii=16
while ~strcmp(maximum,'#Maximum')
  ii=ii+1;
maximum=textread(fichier,'%s',1,'headerlines',ii);
end

fid=fopen(fichier,'r');
ZZZ=cell2mat(textscan(fid,'%f %f %f','headerlines',ii+1));
fclose(fid);

%Output in velocity (m/s)
%400 V/(m/s) is the transfert function of Lennartz 3D5s. sensor 
%2000 V/(m/s) is the transfert function of CMG40TD. sensor 
gain 
fact

ZZZ=ZZZ./((gain*fact));
