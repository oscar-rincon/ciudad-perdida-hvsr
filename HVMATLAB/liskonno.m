
%smoothing of a function y (equally-spaced, dx) with the "Konno-Ohmachi"
%function sin (log10(f/fc)^bexp) / log10(f/fc)^bexp) ^^4
%where fc is the frequency around which the smoothing is performed
%bexp determines the exponent 10^(1/bexp) is the half-width of the peak
%exemple bexp=10,20,30
%cf Konne & Ohmachi, 1998, BSSA 88-1, pp. 228-241



%if nx>16385
%  disp 'echantillon trop grand (max 16385)'
%  return
%end
%lissage script C.michel

% function [ylis]=liskonno(f,y,bexp)
% nx=length(y)
% dx=f(2)-f(1);
% fratio=10^(2.5/bexp);
% ylis(1)=y(1);
% for ix=2:nx
%   fc=f(ix);
%   ix1=floor(fc/fratio/dx);
%   ix2=floor(fc*fratio/dx+1);
%   if ix1<=1
%     ix1=2;
%   end
%   if ix2>=nx
%     ix2=nx;
%   end
%   a1=0;
%   a2=0;
%   for j=ix1:ix2
%     if j~=ix
%       c1=(log10(f(j)/fc))*bexp;
%       c1=(sin(c1)/c1)^4;
%       a2=a2+c1;
%       a1=a1+c1*y(j);
%     else 
%       a2=a2+1;
%       a1=a1+y(ix);
%     end
%   end       
%   ylis(ix)=a1/a2;
% end
% %lissage script HVratio
% 
% function [lfftcvert]=liskonno(f,fftcvert,fftcns,fftceo,lissvar)
% 
% lfftcvert(1,1)=fftcvert(1);
% lfftcvert(1,2)=fftcns(1);
% lfftcvert(1,3)=fftceo(1);
% nt=length(fftcvert);
% dx=f(2)-f(1);
% 			fratio=10^(2.5/lissvar);
% 			for ix=2:fix(nt/2)
%                 clear a1 a2 a3
% 				ifc=ix-1;
% 				xl1=max([ceil(ifc/fratio),1]);
% 				xl2=min([fix(ifc*fratio),nt]);
% 				vliss=lissvar*log10((xl1:xl2)/(ix-1));
% 				vliss(1:(ifc-xl1))=(sin(vliss(1:(ifc-xl1)))./vliss(1:(ifc-xl1))).^4;
% 				vliss((ifc-xl1+2):(xl2-xl1+1))=(sin(vliss((ifc-xl1+2):(xl2-xl1+1)))./vliss((ifc-xl1+2):(xl2-xl1+1))).^4;
% 				vliss(ifc-xl1+1)=1;
%                 a1=sum((fftcvert((xl1+1):(xl2+1)))'.*vliss);
%                 a2=sum((fftcns((xl1+1):(xl2+1)))'.*vliss);
%                 a3=sum((fftceo((xl1+1):(xl2+1)))'.*vliss);
%             
% 				lfftcvert(ix,1)=a1/sum(vliss);
% 				lfftcvert(ix,2)=a2/sum(vliss);
% 				lfftcvert(ix,3)=a3/sum(vliss);
% 			end;
function [lfftcvert]=liskonno(f,fftcvert,fftcns,fftceo,lissvar)

lfftcvert(1,1)=fftcvert(1);
lfftcvert(1,2)=fftcns(1);
lfftcvert(1,3)=fftceo(1);
nt=length(fftcvert);
dx=f(2)-f(1);
			fratio=10^(2.5/lissvar);
			for ix=2:nt
                clear a1 a2 a3
				ifc=ix-1;
				xl1=max([ceil(ifc/fratio),1]);
				xl2=min([fix(ifc*fratio),nt-1]);
                if xl2>=nt-1
                xl2=nt-1;
                end
				vliss=lissvar*log10((xl1:xl2)/(ix-1));
				vliss(1:(ifc-xl1))=(sin(vliss(1:(ifc-xl1)))./vliss(1:(ifc-xl1))).^4;
				vliss((ifc-xl1+2):(xl2-xl1+1))=(sin(vliss((ifc-xl1+2):(xl2-xl1+1)))./vliss((ifc-xl1+2):(xl2-xl1+1))).^4;
				vliss(ifc-xl1+1)=1;
                a1=sum((fftcvert((xl1+1):(xl2+1)))'.*vliss);
                a2=sum((fftcns((xl1+1):(xl2+1)))'.*vliss);
                a3=sum((fftceo((xl1+1):(xl2+1)))'.*vliss);
            
				lfftcvert(ix,1)=a1/sum(vliss);
				lfftcvert(ix,2)=a2/sum(vliss);
				lfftcvert(ix,3)=a3/sum(vliss);
			end;