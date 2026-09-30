
% Calcul du rapportH/V pour chaque fenetre sélectionnées dans trigger
% puis moyenne et écart type géométrique

function [HV_data,f0,A0, f0mean1, f0mean2, HV, f0w1, A0w1, f0w2, A0w2,f01, A01, f02, A02,f0std1 ] = calculHV(Data,fen,tvar,lis_typ,lis_var,Fs,fsensor,fmax)

disp('Calcul H/V')

[o,nbf]=size(fen); %calcul nb de fenetre
[nt,nbsens]=size(Data);

for i=1:nbf
    lf  = fen(2,i)-fen(1,i); %nombre de point de la fenetre de la fenetre
    lf2(i) = 2^nextpow2(lf);
end

L  = max(lf2);

df = 1/(L/Fs);

% réduction des calculs à la fréquence maximale désirée si fmax est bien
% inf à Fs/2.

if fmax<=Fs/2
f  = df:df:fmax;
else
f  = df:df:Fs/2;
end

c = find(f>=fsensor);

%-------I-calcul des FFT de chaque fenetre----------------------------------


for i=1:nbf
    
        for j=1:nbsens
            clear s
            s                         = Data(fen(1,i):fen(2,i),j);
            
            s                            = s - mean(s);
            ws                           = hanning(length(s));
            nw                           = floor(0.02*length(s));
            s(1:nw)                      = s(1:nw) .* ws(1:nw);
            s(length(s):-1:length(s)-nw) = s(length(s):-1:length(s)-nw) .* ...
                                            ws(length(s):-1:length(s)-nw);
            if length(s)<L        
            s(length(s)+1:L)          = 0;     
            end
      
            
            fft2(:,j,i) = log10(abs(fft(s,L))); % calcul de fft jusqu'a Fs/2
            fft1(:,j,i) = fft2(1:length(f),j,i); %réduction de la fft jusqu'a length(f)
        end
        
       
  
%-------II-lissage des FFT de chaque fenetre----------------------------------

        if lis_typ==1 %Konno-Ohmachi lissage

            fft_l(:,:,i) = liskonno(f,fft1(:,1,i),fft1(:,2,i),fft1(:,3,i),lis_var);

        end
            fft_l(:,:,i)=10.^ fft_l(:,:,i);
             
%-------III-clacul des rapports H/V pour chaque fenetre--------------------------------

        rho(:,i)  = sqrt((fft_l(:,2,i).^2+fft_l(:,3,i).^2)./2);%moyenne quadratique
		HVns(:,i) = log10(fft_l(:,2,i)./fft_l(:,1,i));
        HVeo(:,i) = log10(fft_l(:,3,i)./fft_l(:,1,i));
        HV(:,i)   = log10(rho(:,i)./fft_l(:,1,i));

%-------IV-valeur du pic f0 et A0 par fenetres selectionées et écart type et moyenne-----------------------------


end

[nf,n]=size(HV);

%-------IV-clacul des moyenne et écart tyoe-----------------------------


HVmean_log = HV(:,1);
HVmean_logns = HVns(:,1);
HVmean_logeo = HVeo(:,1);
    for j=2:nbf
    HVmean_log   = HVmean_log+HV(:,j);
    HVmean_logns = HVmean_logns+HVns(:,j);
    HVmean_logeo = HVmean_logeo+HVeo(:,j);
    end
HV_mean_log = HVmean_log./nbf;
HV_mean_logns = HVmean_logns./nbf;
HV_mean_logeo = HVmean_logeo./nbf;

HVmean     = 10.^(HV_mean_log);
HVmeanns     = 10.^(HV_mean_logns);
HVmeaneo     = 10.^(HV_mean_logeo);



HV_sig2 = (HV(:,1)-HV_mean_log).^2;
HV_sig2ns = (HVns(:,1)-HV_mean_logns).^2;
HV_sig2eo = (HVeo(:,1)-HV_mean_logeo).^2;
    for j=2:nbf
    HV_sig2 = HV_sig2+(HV(:,j)-HV_mean_log).^2;
    HV_sig2ns = HV_sig2ns+(HVns(:,j)-HV_mean_logns).^2;
    HV_sig2eo = HV_sig2eo+(HVeo(:,j)-HV_mean_logeo).^2;
    end
HV_sig     = sqrt(HV_sig2/(nbf-1));
HV_signs     = sqrt(HV_sig2ns/(nbf-1));
HV_sigeo     = sqrt(HV_sig2eo/(nbf-1));

HV_up      = 10.^(HV_mean_log+HV_sig);
HV_upns      = 10.^(HV_mean_logns+HV_signs);
HV_upeo      = 10.^(HV_mean_logeo+HV_sigeo);

HV_lo      = 10.^(HV_mean_log-HV_sig);
HV_lons      = 10.^(HV_mean_logns-HV_signs);
HV_loeo      = 10.^(HV_mean_logeo-HV_sigeo);

%matrice des résultats: frequence, HV mean tot -sig +sig idem par
%composante d'abord ns puis eo

HV_data(:,1) = f;

HV_data(:,2) = HVmean;
HV_data(:,3) = HV_lo;
HV_data(:,4) = HV_up;

HV_data(:,5) = HVmeanns;
HV_data(:,6) = HV_lons;
HV_data(:,7) = HV_upns;

HV_data(:,8) = HVmeaneo;
HV_data(:,9) = HV_loeo;
HV_data(:,10) = HV_upeo;

% trouver pic sur courbe H/V moyenne


[A0(1),z(1)]=max(HVmean(c(1)+round(0.05/df):nf));
[A0(2),z(2)]=max(HV_lo(c(1)+round(0.9*z(1)):c(1)+round(1.1*z(1))));
[A0(3),z(3)]=max(HV_up(c(1)+0.9*z(1):c(1)+1.1*z(1)));
f0(1) = f(z(1)+c(1)+round(0.05/df)-1);
f0(2) =f(z(2)+c(1)+round(0.9*z(1))-1);
f0(3) = f(z(3)+c(1)+round(0.9*z(1))-1);

% pointage autre 
table=xlsread('ttest-table.xls');
[f01 A01 f02 A02]=pointage_f0_std(10.^HV_mean_log,HV_sig,f,table,nbf.*ones(length(f),1),2)

% trouver pic sur chaque fenetre
w=0;
for i =1:nbf
    [A0w22,I]=max(HV(z(1)+c(1)-round((0.2*f01)/df):nf,i));
    %0.2 critere 8 SESAM on pick les fréquences max
    % dans une gamme de fréquence proche du pic de la courbe moyenne (en basse fréquence uniquement, pour éviter le bruit).

    a=I+z(1)+c(1)-round((0.2*f01)/df)-1;
    if a>length(f)
        a=length(f);
    end

    
%    if I<z(1)+c(1)+round((2*f0(1))/df) si désiré dans une gamme de fréquence
%    proche du pic de la courbe moyenne (haute fréquence).

    w=w+1;
    f0w1(w) = f(a);
    A0w1(w) = A0w22;
 %   end
end

% trouver pic sur chaque fenetre
w=0;
for i =1:nbf
    [A0w22,I]=max(HV(z(1)+c(1)-round((0.2*f02)/df):nf,i));
    %0.2 critere 8 SESAM on pick les fréquences max
    % dans une gamme de fréquence proche du pic de la courbe moyenne (en basse fréquence uniquement, pour éviter le bruit).

    a=I+z(1)+c(1)-round((0.2*f02)/df)-1;
    if a>length(f)
        a=length(f);
    end

    
%    if I<z(1)+c(1)+round((2*f0(1))/df) si désiré dans une gamme de fréquence
%    proche du pic de la courbe moyenne (haute fréquence).

    w=w+1;
    f0w2(w) = f(a);
    A0w2(w) = A0w22;
 %   end
end

%calcul moyenne des fréquences et amplitude des pics de chaque fenetre

f0mean1_log     = mean(log10(f0w1));
f0mean1     = 10.^f0mean1_log;
f0std1     = std(log10(f0w1));

f0mean2_log     = mean(log10(f0w2));
f0mean2     = 10.^f0mean2_log;
f0std2     = std(log10(f0w2));


