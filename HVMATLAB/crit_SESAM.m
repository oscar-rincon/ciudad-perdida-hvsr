%---------------------------------------------------------------------
% script permettant de vérifier les critères SESAM. J.regnier avril 2009
%-------------------------------------------------------------------------

function crit=crit_SESAM(HV_data, fen, tmin, f0, A0, f0_sig,fsensor)
disp('vérifie critere SESAME')

%paramêtres à rentrer: longueur des denêtres et fréquence du capteur
lw = tmin;

[o,nw] = size(fen); %calcul nb de fenetre


% f : fréquene, std écartype sur l'amplitude
f      = HV_data(:,1);
HVmean = HV_data(:,2);
HV_lo  = HV_data(:,3);
HV_up  = HV_data(:,4);
std    = log10(HV_up./HVmean); 
df     = abs(f(1)-f(2));

%max dans H/V mean et Plus ou Moins écartype provient de la fonction
%trouver_pic

A02   = A0(1);     %max de la courbe moyenne du rapport H/V
A0pl  = A0(2);
A0mo  = A0(3);
f02   = f0(1);      %max de la courbe moyenne du rapport H/V
f0pl  = f0(2);
f0mo  = f0(3);
stdf  = f0_sig ;   % écart type sur la fréquence du pic dans chacune des fenetres

n0    = find((f02-df)< f & f <(f02+df));
% Critères seuils epsilon et teta

E(1)    = 0.2*f02;
teta(1) = 2.5;
E(2)    = 0.15*f02;
teta(2) = 2;
E(3)    = 0.1*f02;
teta(3) = 1.78;
E(4)    = 0.05*f02;
teta(4) = 1.58;

%script to check the reliability of H/V curve

%critere 1__________________________________
if f02>10/lw
    crit1=1;
else 
    crit1=0;
end
%critere 2__________________________________
if nw*lw*f02>200
    crit2=1;
else
    crit2=0;
end
%critere 3__________________________________
I = find(0.5*f02 < f & f < 2*f02);

crit3=1;
for n=1:length(I)
    if std(I(n))<2
    crit3=1*crit3;
    else
    crit3=0*crit3;
    end
end

%script to check the peak H/V______________

clear I J
%critere 4_________________________________
I = find(0.25*f02 < f & f < f02);
J = find(f02 < f & f < 4*f02);

crit4=0;
for n=1:length(I)
    if HVmean(I(n))<A02/2
    crit4=1+crit4;
    else
    crit4=0+crit4;
    end
end
if crit4>0
    crit4=crit4/crit4;
else
    crit4=0;
end
%critere 5_________________________________
crit5=0;
for n=1:length(J)
   if HVmean(J(n))<A02/2
    crit5=1+crit5;
   else
    crit5=0+crit5;
end
end

if crit5>0
    crit5=crit5/crit5;
else
    crit5=0;
end
%critere 6_________________________________
if A02>2
    crit6=1;
else
    crit6=0;
end

%critere 7_________________________________


crit7=1

if (f02*0.95 <= f0(2) & f0(2) <= f02*1.05 & f02*0.95 <= f0(3) & f0(3) <= f02*1.05)
    crit7=1
else
    crit7=0*crit7
end


%critere 8 et 9__________________________

if 0.2 < f02 & f02 <= 0.5
    if  std(n0)<teta(1)
        crit9=1;
    else
        crit9=0;
    end
    if  stdf<E(1)
        crit8=1;
    else
        crit8=0;
    end
elseif 0.5 < f02 & f02 <= 1
    if  std(n0)<teta(2)
        crit9=1;
    else 
        crit9=0;
    end
    if  stdf<E(2)
        crit8=1;
    else
        crit8=0;
    end
elseif 1 < f02 & f02 <= 2
     if  std(n0)<teta(3)
        crit9=1;
     else 
        crit9=0;
     end
        if  stdf<E(3)
        crit8=1;
        else
        crit8=0;
        end
elseif f02 > 2
     if  std(n0)<teta(4)
        crit9=1;
     else 
        crit9=0;
     end
        if  stdf<E(4)
        crit8=1;
        else
        crit8=0;
        end
end


crit(1)=crit1;
crit(2)=crit2;
crit(3)=crit3;
crit(4)=crit4;
crit(5)=crit5;
crit(6)=crit6;
crit(7)=crit7;
crit(8)=crit8;
crit(9)=crit9;



% name=sprintf('HV_sesam');
% 
% FID = fopen(name,'w');
% 
% fprintf(FID,'%s\n',name);
% fprintf(FID,'longueur de la fenêtre: %f s\n',lw);
% fprintf(FID,'nombe de fenêtres: %d\n',nw);
% fprintf(FID,'frequence du capteur: %f_Hz\n',fsensor);
% fprintf(FID,'frequence du pic: %f Hz\n',f02);
% fprintf(FID,'crit1 crit2 crit3 crit4 crit5 crit6 crit7 crit8 crit9 \n%d %d %d %d %d %d %d %d %d\ ',crit1, crit2, crit3, crit4, crit5 , crit6, crit7, crit8, crit9);
%  fclose(FID);
f0