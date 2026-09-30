
% fonction pointage de f0

% H.Cadet et J.regnier 2011

% fBTF vercteur fréquence
% BTF moyenne des fonctions d'amplifications
% std Žcart type en Žchelle log10
% si pas de pics net ne vérifiant pas le ttest alors f0 et A0 seont nuls.
% AMP amplitde à laquelle le t-test est réalisé

function [f0 A0 fmax Amax]=pointage_f0(BTF,std,fBTF,table,ns,AMP)


%Trouver f0 et fmax

a=isnan(BTF(:,1));
b=find(a==0);
c=find(fBTF>0.2);

aTF = BTF(c(1):end,1);
f   = fBTF((c(1):end));

[maxtab, mintab]=peakdet(aTF, 0.5);

if isempty(maxtab)
    f0 = 0;
    A0 = 0;
    fmax = 0;
    Amax = 0;
else

    [g g1] = size(maxtab);

    for k=1:g
    end
    [e ff]=max(maxtab(:,2));

    Amax = e;
    fmax = f(maxtab(ff,1));

    %one sample t-test sur ces maximaux locaux

    if max(ns)>1
        alfa = 0.005;
        colo = 6;
    else
        alfa = 0.025;
        colo = 4;
    end

  
    if g<2
        f0 = f(maxtab(ff(1),1));
        A0 = e(1);
    else

        for n=1:g
            
            %one sample t-test sur ces maximaux locaux
            
            testt(n)= (log10(maxtab(n,2))-log10(AMP)) / ((std(maxtab(n,1)))/ (sqrt(ns(maxtab(n,1))+1)));
            
            if ns(maxtab(n,1))<99
                if testt(n)>=table(ns(maxtab(n,1))+3,colo)
                    h=1;
                else
                    h=0;
                end
                test2(n)= h;
            else
                if testt(n)>=table(101,colo)
                    h=1;
                else
                    h=0;
                end
                test2(n)= h;
            end
        end
        for n=1:g
           
            if  test2(n)==1
                f0 = f(maxtab(n,1));
                A0 = maxtab(n,2);
                break
            else
                f0 = 0;
                A0 = 0;
            end
        end

    end


end

