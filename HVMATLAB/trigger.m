
%--------------------------------------------------------------------------
% trigger pour le H/V (triggera.m et PSD_mat.m) modifier J.régnier avril 2009
%--------------------------------------------------------------------------


function [fen, datamoy, lta_sta]=trigger(data,lta,sta,lta_sta_min,lta_sta_max,Fs,tmin,tmax,tvar,overlap,trig_var)

disp('choisi les fenetres de bruit stationnaire');
%est  = data(:,3);
%nord = data(:,2);%attention vérifier que les composantes sont bien sur la bonne voie
%vert = data(:,1);

[nt,nbsens]=size(data);
dt=1/Fs;
t=dt:dt:(dt*nt);
nlta = round(lta*Fs); % nombre de points de la lta
nsta = round(sta*Fs); % nombre de points de la sta

nmin = tmin*Fs;
nmax = tmax*Fs;

teste=ones(nt,1);

for i=1:nbsens %boucle sur les trois composantes

%I-apodisation du signal sur 2% avec une fenêtre hanning afin de supprimer la composante continue
%des vitesses du sol engendrée par l'appareil de mesure---------------------

    s                            = data(:,i);
    s                            = s - mean(s);
    ws                           = hanning(length(s));
    nw                           = floor(0.02*length(s));
    s(1:nw)                      = s(1:nw) .* ws(1:nw);
    s(length(s):-1:length(s)-nw) = s(length(s):-1:length(s)-nw) .* ...
                                   ws(length(s):-1:length(s)-nw);
    data_a(:,i)                  = s;           

size(data_a);
end


datamoy = sqrt(data_a(:,1).^2+data_a(:,2).^2+data_a(:,3).^2);
datamoymoy = mean(datamoy);
maxim2 = max(datamoy);


%II-Clacul de LTA et STA-------------------------------------------------------
   
   if trig_var==0
       for i=1:nbsens 
        clear sigmoy
        sigmoy = abs(data_a(:,i));
       i0=nlta+1;
        maxim  = max( sigmoy );
        lta    = calculta(sigmoy, nlta, datamoymoy,i0); %lta et sta sont ici des vecteurs
        sta    = calculta(sigmoy, nsta, datamoymoy,i0);
        lta_sta(i)=sta./lta;
        teste=teste.*((lta_sta(i)>lta_sta_min)&(lta_sta(i)<lta_sta_max)&(sigmoy < (maxim*.99)));
       
       end
   else
         i0=nlta+1;
         lta    = calculta(datamoy, nlta, datamoymoy,i0); %lta et sta sont ici des vecteurs
         sta    = calculta(datamoy, nsta, datamoymoy,i0);
         lta_sta=sta./lta;
         % 
         % figure
         % subplot(2,1,1)
         % plot(t,(lta./sta),'-b')
         % axis([0 t(end) -4 4])
         % hold on
         % plot(t,2,'g')
%          subplot(3,1,2)
%          plot(t,lta,'k')
%          hold on
%          plot(t,sta,'m')
          teste=teste.*((lta_sta>lta_sta_min)&(lta_sta<lta_sta_max)&(datamoy < (maxim2*.99)));
    end

%III-selection des fenetres respectrant le LTA/STA------------------------

 %   subplot(2,1,2)
 %  plot(t,datamoy,'r')
 %  hold on
 % plot(t,lta_sta_max,'r')
%   
 

nbwin=0;
fen=[];

io=1;  
 
%if tvar==1 %fenêtre pas fixe

 if tvar==0
     nmax=nmin;
 end
    fenetre=[];
    while (io+nmax-1)<length(data)
        z=1;
            iinit=io;
            ifinal=iinit+nmax;
            nlongmax=nmin;
	        it=1;
            while (prod(teste(int16(iinit):int16(ifinal))))==0
		           ifinal=ifinal-1;
                if (ifinal-iinit)<nmin
                    z=0;
                    break
                end
                
            end

        if z==1
         
            fenetre=[iinit;ifinal];
            noverlap = round(overlap/100*(ifinal-iinit));
            io=io+ifinal-iinit-noverlap;
             % plot(t(iinit:ifinal),maxim2,'k','linewidth',2.5)
             % hold on
             % plot(t(iinit:ifinal),-maxim2,'k','linewidth',2.5)
             % plot(t(iinit:(iinit+1)),[-maxim2;maxim2],'k','linewidth',2.5)
             % plot(t((ifinal-1):ifinal),[maxim2;-maxim2],'k','linewidth',2.5)
            fen=[fen,fenetre];
        else
            noverlap = round(overlap/100*(ifinal-iinit));
            W=find((teste(iinit:ifinal+1))==0);
            io=io+W(length(W))+1;
	    end
	
    end
 
    

% ylabel(' HVNSR amplitude','FontSize',24,'FontWeight','bold')
% xlabel(' signal (s)','FontSize',24,'FontWeight','bold')



