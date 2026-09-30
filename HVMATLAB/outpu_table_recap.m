%script permettat décrire un fichier récapitulatif sur les donées traitées

clear all
close all

%--------------------------------------------------------
% I- Lecture du fichier d'entrée des paramètres: input.txt
%--------------------------------------------------------


% [input,PathName] = uigetfile({'*.txt*'},'Select the inputfile');
% input.txt=[char(PathName),input];


input.txt='input.txt';


[lta com]       = textread(input.txt,'%f %s',1,'headerlines',0);
[sta com]       = textread(input.txt,'%f %s',1,'headerlines',1);
[seuilmin com]  = textread(input.txt,'%f %s',1,'headerlines',2);
[seuilmax com]  = textread(input.txt,'%f %s',1,'headerlines',3);
[tmin com1]      = textread(input.txt,'%f %s',1,'headerlines',4) ;%longeur de fenètre minimale
[tvar com]      = textread(input.txt,'%f %s',1,'headerlines',5) ;%longeur de fenetre variable 0 non 1 oui
[tmax com]      = textread(input.txt,'%f %s',1,'headerlines',6);
[overlap com]   = textread(input.txt,'%f %s',1,'headerlines',7); %recouvrement de fenètres en %
[lis_typ com]   = textread(input.txt,'%f %s',1,'headerlines',8); %lis_typ = 1 kono
[lis_var com]   = textread(input.txt,'%f %s',1,'headerlines',9); %paramèrtre de lissage b=10 20 30 40 allant du plus lissé au moins lissé
[trig_var com]   = textread(input.txt,'%f %s',1,'headerlines',10); % parametre permettant de savoir si le trigger est fait sur la moyenne quadratique des composantes ou sur les 3 composantes simulatnélent
[fsensor com]   = textread(input.txt,'%f %s',1,'headerlines',11);
[deci com]   = textread(input.txt,'%f %s',1,'headerlines',12);%donne le taux de décimation du signal si souhaité: 1 tt les points 2 tout les 2 points...
[fmax com]   = textread(input.txt,'%f %s',1,'headerlines',13);%fréquence maximale pour le fichier output, fmax doit etre inf a Fs/deci


%--------------------------------------------------------
% II- Lecture des fichiers d'entrée des données: décrits 
%     par une liste et contenus dans path_data
%--------------------------------------------------------



[F]=textread('liste_nice','%s');
B=F;
path_data = '/Users/JR/Documents/travail-2010/BDF-nice/BDF1/';
path_result = '/Users/JR/Documents/travail-2010/BDF-nice/BDF1/';
l=size(B);

%--------------------------------------------------------
% III- Boucle sur les fichiers de donnée 
%--------------------------------------------------------

fid  = fopen('recap_nice_BDF.txt','w');

fprintf(fid,'%s %g\n','lta',lta);
fprintf(fid,'%s %g\n','sta',sta);
fprintf(fid,'%s %g\n','sta/lta_min',seuilmin);
fprintf(fid,'%s %g\n','sta/lta_max',seuilmax);
fprintf(fid,'%s %g\n','fenetre_min',tmin);
if tvar==0
fprintf(fid,'%s %s\n','fenetre_fixe','y');
else
fprintf(fid,'%s %s %s %g\n','fenetre_fixe','non','fenetre_max',tmax);
end
fprintf(fid,'%s %s\n','type_lissage','okono');
fprintf(fid,'%s %g\n','b',lis_var);
fprintf(fid,'%s %g\n','f_sensor',fsensor);
fprintf(fid,'%s %s %s %s %s %s %s %s %s %s %s %s %s\n','name mesure','nbwin','f0mean','A0mean','C1','C2','C3','C4','C5','C6','C7','C8','C9');

for j=1:10 
        clear C name2 Data

            
            name_t=char(B((j-1)*3+1));
            name_ti=name_t(1:10);
            

toto = sprintf ('HV_%s.txt',name_ti);

tot=[char(path_data),toto];

[nbwin nw]       = textread(tot,'%s %f',1,'headerlines',9);
[f0 f0m f0pl]       = textread(tot,'%f %f %f',1,'headerlines',13);
[A0 A0m A0pl]  = textread(tot,'%f %f %f',1,'headerlines',15);
[C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8) C(9)]  = textread(tot,'%f %f %f %f %f %f %f %f %f',1,'headerlines',17);

fprintf(fid,'%s %g %f %f %g %g %g %g %g %g %g %g %g\n',name_ti,nw,f0,A0,C(1),C(2),C(3),C(4),C(5),C(6),C(7),C(8),C(9));

end
  fclose(fid);
  
    