

function [ZZZ,Fs,ncap]=load_city(fichier,path,deci)
%[ZZZ,Fs]=load_city(fichier)
%Opens a Cityshark II file and gives a matrix ZZZ with nt rows (number of samples), ncomp columns (number of components) 
%and the sampling frequency Fs. ZZZ is in m/s, gain is corrected, transfert
%function is for Le3D5s.
disp('charge le fichier')
%Controls the number of headerlines that varies depending on the Readcity version.
fichier=[char(path),fichier];
[station]=textread(fichier,'%s',1,'headerlines',1);
if strcmp(station,'Station')
  plus=0;
else
  plus=2;
end
[chan nb ncomp]=textread(fichier,'%s %s %f',1,'headerlines',3+plus);
try
  [samp ra Fs]=textread(fichier,'%s %s %f',1,'headerlines',8+plus);
  [samp nb nt]=textread(fichier,'%s %s %f',1,'headerlines',9+plus);
  [co factor fact]=textread(fichier,'%s %s %f',1,'headerlines',11+plus);
  [ga gain]=textread(fichier,'%s %f',1,'headerlines',12+plus);
catch
  [samp ra Fs]=textread(fichier,'%s %s %f',1,'headerlines',6+plus);
[samp nb nt]=textread(fichier,'%s %s %f',1,'headerlines',7+plus);
[co factor fact]=textread(fichier,'%s %s %f',1,'headerlines',9+plus);
[ga gain]=textread(fichier,'%s %f',1,'headerlines',10+plus);
end
maximum=[];
ii=10+plus;
while ~strcmp(maximum,'Maximum')
  ii=ii+1;
maximum=textread(fichier,'%s',1,'headerlines',ii);
end

chaine='%f ';
for i=1:ncomp-1
  chaine=[chaine '%f '];
end
fid=fopen(fichier,'r');
ZZ=textscan(fid,chaine,nt,'headerlines',ii+1);
ZZZ2=cell2mat(ZZ);
fclose(fid);

%Output in velocity (m/s)
%400 V/(m/s) is the transfert function of Lennartz 3D5s. sensor 

ZZZ2=ZZZ2/(gain*fact)*1/400;

for i=1:ncomp
  ZZZ(:,i)=decimate(ZZZ2(:,i),deci);
end
ncap=ncomp/3;
Fs=Fs/deci;
% Fs=Fs/deci;
% ZZZ(1,:)=ZZZ2(1,:);
% for j=2:round(length(ZZZ2(:,1))/deci)
%     ZZZ(j,:)=ZZZ2((deci*(j-1)),:);
% end
% size(ZZZ2)
% size(ZZZ)