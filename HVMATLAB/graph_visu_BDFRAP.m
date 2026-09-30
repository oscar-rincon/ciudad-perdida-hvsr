% script pour plotter sur une page un rapport HV un graph avec les
% fenetres selectionnées et le tableau des critères SESAM J.regnier avril
% 2009 modifications Mars 2010


res=get(0,'ScreenSize');
figure('Position',[1 1 res(3) res(4)-64])
disp('trace graphique')
f = HV_data(:,1);

HVmean   = HV_data(:,2);
HV_lo    = HV_data(:,3);
HV_up    = HV_data(:,4);

HVmeanns = HV_data(:,5);
HV_lons  = HV_data(:,6);
HV_upns  = HV_data(:,7);

HVmeaneo = HV_data(:,8);
HV_loeo  = HV_data(:,9);
HV_upeo  = HV_data(:,10);

c = find(f>=fsensor);


% premier subplot montant le rapport H/V----------------------------

subplot('Position',[0.1 0.4 0.8 0.53])

semilogx(f,HVmean, 'r', 'linewidth', 3.5)
hold on
semilogx(f,HV_up, '--r', 'linewidth', 1.5)
semilogx(f,HV_lo, '--r', 'linewidth', 1.5)

%peut tracer le H/V individuel par fenètres 

%semilogx(f,10.^HV,'linewidth', 1)

h1=semilogx(f0w1,10.^A0w1,'Color',[0 0.8 0.2],'LineStyle','none','Marker','x', 'markersize', 12,'linewidth',2);

%peut plotter l'erreur sur f0 et A0

% LX=10.^(log10(f0mean)-f0_sig);
% UX=10.^(log10(f0mean)+f0_sig);
% LX1=10.^(log10(A0mean)-A0_sig);
% UX1=10.^(log10(A0mean)+A0_sig);
%annotation('doublearrow',
% h=ploterr(f0mean,A0mean,{LX,UX},{LX1,UX1});
% set(h(2),'MarkerSize',20,'LineWidth',1.5,...
%     'LineStyle','-',...
%     'Color',[0 0.4 1])
% set(h(3),'MarkerSize',20,'LineWidth',1.5,...
%     'LineStyle','-',...
%     'Color',[0 0.8 0.2])
% legend([h1(1), h(3)], 'Max on indiviual window','16-84th percentile f_0 per window')


legend([h1(1)], 'Max on indiviual window')

grid

%tit= sprintf('H/V %s capteur %g', char(B((j-1)*3+l)), w);

%tit = strrep(tit, '_', ' ');
%tit = strrep(tit, '\', ' ');
tit = strrep(name, '_', ' ');
title (tit,'FontSize',20,'FontWeight','bold')
ylabel(' HVNSR amplitude','FontSize',16,'FontWeight','bold')
xlabel(' Frequency (Hz)','FontSize',16,'FontWeight','bold')
F=roundn(f0(1),-2);
F2=roundn(A0(1),-2);

a= sort([0.2:0.2:1 2:2:10 20 30 50 F]);

A2=find (a==F);
if length(A2)>1
    clear a
    a= sort([0.2:0.2:1 2:2:10 20 30 50]);
end

b=sort([0 1 2 3 4 5 6 7 8 9 10 F2]);
B2=find (b==F2);
if length(B2)>1
    clear b
    b= sort([0 1 2 3 4 5 6 7 8 9 10]);
end
axis([fsensor fmax 0 max(HV_up(c(1):end))+1])
set(gca,'FontSize',12,'FontWeight','bold','XGrid','on','XTick',a,'YGrid','on','YMinorGrid','off',...
    'YTick',b)


% deuxieme subplot montant les fentres du signal selectionnées---------

[o,nbf]=size(fen);
[nt,nbsens]=size(Data);
dt=1/Fs;
t=dt:dt:(dt*nt);
s = Data(:,2);
maxim=max(s(1000:end-100));

subplot('Position',[0.1 0.05 0.65 0.20])
plot(t,s,'r')
hold on

for i= 1:nbf
iinit  = round(fen(1,i));
ifinal = round(fen(2,i));
plot([t(iinit) t(ifinal) t(ifinal) t(iinit) t(iinit) ],[0.9*maxim 0.9*maxim -0.9*maxim -0.9*maxim 0.9*maxim],'k','linewidth',1.5)
end
%ylabel(' mean signal (m/s)','FontSize',12,'FontWeight','bold')
ylabel('  N-S signal (10^-^6 Volts)','FontSize',12,'FontWeight','bold')

tit2= sprintf('windows selected');

tit2 = strrep(tit2, '_', ' ');
tit2 = strrep(tit2, '\', ' ');
title (tit2,'FontSize',16,'FontWeight','bold')      
axis([0 t(end) -maxim maxim])
set(gca,'FontSize',12,'FontWeight','bold','XGrid','on','YGrid','on','YMinorGrid','off')


% troisieme subplot montrant les criteres sesame------------------------

subplot('Position',[0.8 0.05 0.15 0.2])

 plot(0.4:0.5:0.8,0); 
 ylim([0 1])
 xlim([0.1 0.8])
 cri=['C1';'C2';'C3';'C4';'C5';'C6';'C7';'C8';'C9'];
 for o=1:9
 cr=sprintf('%s',cri(o,:));
     text(0.2,1-0.1*o,cr,'HorizontalAlignment','center', 'color',[0.2,0.6,0.5],'FontSize',12,'FontWeight','bold')
     if crit(o)==0
     text(0.6,1-0.1*o,'NO','HorizontalAlignment','center', 'color',[1,0,0],'FontSize',12,'FontWeight','bold')
     else
     text(0.6,1-0.1*o,'OK', 'HorizontalAlignment','center','color',[0,0,0],'FontSize',12,'FontWeight','bold')
     end
 end
 name1=sprintf('f_0 = %5.3g Hz',f0(1));
 name3=sprintf('A_0 = %5.3g', A0(1));
 text(0,1.4,name1, 'color',[0 0.8 0.2],'FontSize',14)
 text(0.5,1.4,name3, 'color',[0 0.4 1],'FontSize',14)

 set(gca,'FontSize',9,'FontWeight','bold','XGrid','off','YGrid','off','XTick',[],'XTickLabel',{''},'YTickLabel',{''})
title ('SESAME Criteria','FontSize',16,'FontWeight','bold')

%enregistremet de la figure-----------------------------------------------

set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [2 1 12 10]);

%toto=sprintf ('/HV_%s_capt%g.jpg',char(B((j-1)*3+l)),w);
toto=name(1:end-4);
saveas(gcf,toto);
