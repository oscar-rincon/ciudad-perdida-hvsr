

function [ZZZ,Fs]=load_ASCII(fichier,path,deci)

%Opens a ASCII file  4 columns and gives a matrix ZZZ with nt rows (number of samples), ncomp columns (number of components) 
%and the sampling frequency Fs. ZZZ is in m/s, gain is corrected, transfert
%function is for Le3D5s.
disp('charge le fichier')
%Controls the number of headerlines that varies depending on the Readcity version.

fichier=load([path,fichier]);

t2=fichier(:,1);

Fs=1/(t2(2)-t2(1));


ZZZ(:,1)=decimate(fichier(:,4)./1000,deci);
ZZZ(:,2)=decimate(fichier(:,2)./1000,deci);
ZZZ(:,3)=decimate(fichier(:,3)./1000,deci);
Fs=Fs/deci;

