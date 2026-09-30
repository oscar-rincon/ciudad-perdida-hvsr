%------------------------------------------------------------------------

%fonction permettant de calculter LTA et STA sig_moy données 
%et ita nb de points pour le calcul (ici nlta ou nsta)
%les points avant nlta sont pris égaux à la moyenne de la moyenne
%quadratique des 3 composantes. 

function ta = calculta(sig_moy, ita, datamoymoy,i0)
	
nt=length(sig_moy);
ta=zeros(nt,1);
ta(1:ita-1)=datamoymoy; 
ta(i0-1) = mean(sig_moy(i0-ita:i0-1));
for i=i0:(nt)	   
    ta(i) = ta(i-1) - sig_moy(i-ita)/ita + sig_moy(i)/ita;	
end

% function ta = calculta(sig_moy, ita, datamoymoy)
% 	
% nt=length(sig_moy);
% ta=zeros(nt,1);
%  
% ta(1) = mean(sig_moy(1:ita));
% for i=2:nt-ita	   
%     ta(i) = ta(i-1) - sig_moy(i-1)/ita + sig_moy(i+ita)/ita;	
% end
% ta(nt-ita+1:nt)=datamoymoy;