% script permettant d'enregistrer les résultats
% le fichier sortie se compose d'un hearders dans lequel sont récapitulés
% les paramètres d'analyse, le pic, les criteres SESAM et la moyenne et percentiles du H/V

[o,nw] = size(fen);
f = HV_data(:,1);
df=f(2)-f(1);
HVmean   = HV_data(:,2);
HV_lo    = HV_data(:,3);
HV_up    = HV_data(:,4);

 HVmeanns = HV_data(:,5);
 HV_lons  = HV_data(:,6);
 HV_upns  = HV_data(:,7);
 
 HVmeaneo = HV_data(:,8);
 HV_loeo  = HV_data(:,9);
 HV_upeo  = HV_data(:,10);

toto = sprintf ('HV_%s.txt',nam);

fid  = fopen(toto,'w');

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
fprintf(fid,'%s %g\n','nb_win',nw);
fprintf(fid,'%s %g\n','f0_mean_windows',f0mean1);
fprintf(fid,'%s %g\n','f0_sig_windows',f0_sig);
fprintf(fid,'%s %s %s\n','f0mean','f0low','f0up');
fprintf(fid,'%g %g %g\n',f0(1),f0(2),f0(3));
fprintf(fid,'%s %s %s\n','A0mean','A0low','A0up');
fprintf(fid,'%g %g %g\n',A0(1),A0(2),A0(3));
fprintf(fid,'%s %s %s %s %s %s %s %s %s\n','C1','C2','C3','C4','C5','C6','C7','C8','C9');
fprintf(fid,'%g %g %g %g %g %g %g %g %g\n',crit(1),crit(2),crit(3),crit(4),crit(5),crit(6),crit(7),crit(8),crit(9));
i=1;
 fprintf(fid,'%s %s %s %s %s %s %s %s %s %s\n','f(Hz)','HVmean','HV_up','HV_lo','HVmean_ns','HV_up_ns','HV_lo_ns','HVmean_ew','HV_up_ew','HV_lo_ew');
while f(i)<=fmax-df
    fprintf(fid,'%f %f %f %f %f %f %f %f %f %f\n',f(i),HVmean(i),HV_up(i),HV_lo(i),HVmeanns(i),HV_upns(i),HV_lons(i),HVmeaneo(i),HV_upeo(i),HV_loeo(i));
    i = i+1;
end
  fclose(fid);
  
  %ti=sprintf('%s',path_result) 
  %movefile('HV*.txt',char(path_result));
    