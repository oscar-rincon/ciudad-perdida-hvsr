%--------------VERSION MAC----------------------------------------------

% fichier principal de calcul des H/V bruit de fond avec fichier city shark
% en entr�e

clear all
close all

%--------------------------------------------------------
% I- Lecture du fichier d'entr�e des param�tres: input.txt
%--------------------------------------------------------


% [input,PathName] = uigetfile({'*.txt*'},'Select the inputfile');
% input.txt=[char(PathName),input];


input.txt='input.txt';
addpath('./MatSAC/')

[lta com]       = textread(input.txt,'%f %s',1,'headerlines',0);
[sta com]       = textread(input.txt,'%f %s',1,'headerlines',1);
[seuilmin com]  = textread(input.txt,'%f %s',1,'headerlines',2);
[seuilmax com]  = textread(input.txt,'%f %s',1,'headerlines',3);
[tmin com1]      = textread(input.txt,'%f %s',1,'headerlines',4) ;%longeur de fen�tre minimale
[tvar com]      = textread(input.txt,'%f %s',1,'headerlines',5) ;%longeur de fenetre variable 0 non 1 oui
[tmax com]      = textread(input.txt,'%f %s',1,'headerlines',6);
[overlap com]   = textread(input.txt,'%f %s',1,'headerlines',7); %recouvrement de fen�tres en %
[lis_typ com]   = textread(input.txt,'%f %s',1,'headerlines',8); %lis_typ = 1 kono
[lis_var com]   = textread(input.txt,'%f %s',1,'headerlines',9); %param�rtre de lissage b=10 20 30 40 allant du plus liss� au moins liss�
[trig_var com]   = textread(input.txt,'%f %s',1,'headerlines',10); % parametre permettant de savoir si le trigger est fait sur la moyenne quadratique des composantes ou sur les 3 composantes simulatn�lent
[fsensor com]   = textread(input.txt,'%f %s',1,'headerlines',11);
[deci com]   = textread(input.txt,'%f %s',1,'headerlines',12);%donne le taux de d�cimation du signal si souhait�: 1 tt les points 2 tout les 2 points...
[fmax com]   = textread(input.txt,'%f %s',1,'headerlines',13);%fr�quence maximale pour le fichier output, fmax doit etre inf a Fs/deci


%--------------------------------------------------------
% II- Lecture des fichiers d'entr�e des donn�es: d�crits
%     par une liste et contenus dans path_data
%--------------------------------------------------------



%[F]=textread('liste_nice','%s');
% B={'0032_240613_1218.txt';'0032_240613_1238.txt';
% '0032_240613_1258.txt';
% '0037_240613_1232.txt';
% '0037_240613_1300.txt';
% '0037_240613_1349.txt'};
path_data = '/Users/julie/Desktop/HV pour Silvana/';
A=dir([path_data '*.sac']);
B=A.name;%{'160323_1500.164'};
path_result = path_data;

%--------------------------------------------------------
% III- Boucle sur les fichiers de donn�e
%--------------------------------------------------------


for j=1%:length(A)
    clear C name2 Data
    %nam = '0037_250911_1625.txt'
    %fichier = [path_data nam];

    %--------------------------------------------------------
    % LOAD SAC FILES
    %--------------------------------------------------------
    for l=1:3
        name=char(A((j-1)*3+l).name);
        fichier=[char(path_data),char(A((j-1)*3+l).name)];
        [t,Data(:,l),SAChdr]=fget_sac(fichier);
    end

    deltat=t(2)-t(1);
    Fs=1/(deltat); % sampling frequency

    %--------------------------------------------------------
    % LOAD CITYSHARK FILES
    %--------------------------------------------------------

    %         C=B(j);
    %         name2 = char(cellstr(C));
    %         [Data2,Fs,ncap]=load_city(name2,char(path_data),deci);


    %--------------------------------------------------------
    % LOAD ASCII FILES
    %--------------------------------------------------------

    % for l=1:3
    %     name=[char(path_data),char(B((j-1)*3+l))];
    %     [Data(:,l)]=textread(name,'%f','headerlines',1);
    %
    % end
    %     [s2, stat, X, s1, dt, ms, time, T, t, v, i, mi]=textread(name,'%s %s %s %s %f %s %s %s %s %s %s %s',1,'headerlines',0);
    %     Fs=1/(dt*0.001);
    %
    %     name_t=char(B((j-1)*3+l));
    %     name_ti=name_t(1:10);

    %--------------------------------------------------------
    % LOAD Minishark FILES
    %--------------------------------------------------------

    % [Data,Fs,time]=load_minishark(fichier);
    % [Data,Fs,time]=load_minishark_marco(fichier);
    % name=fichier;
    %--------------------------------------------------------
    % A-Trigger: Choix des fenetres stationnaire
    %--------------------------------------------------------

    [fen,datamoy,lta_sta]=trigger(Data,lta,sta,seuilmin,seuilmax,Fs,tmin,tmax,tvar,overlap,trig_var);

    %--------------------------------------------------------
    % B-Calcul H/V et f0 A0
    %--------------------------------------------------------

    [HV_data,f0,A0, f0mean1, f0mean2, HV, f0w1, A0w1, f0w2, A0w2,f01, A01, f02, A02,f0_sig]=calculHV(Data,fen,tvar,lis_typ,lis_var,Fs,fsensor,fmax);

    %--------------------------------------------------------
    % C- V�rification des crit�res SESAME sur le pic s�lectionn�
    % avant
    %--------------------------------------------------------

     crit=crit_SESAM(HV_data, fen, tmin, f0, A0, f0_sig,fsensor);

    %--------------------------------------------------------
    % Plot des r�sultats et tableaux des r�ultats
    %--------------------------------------------------------
    w=1;
    graph_visu_BDFRAP;
    output_table;


end

