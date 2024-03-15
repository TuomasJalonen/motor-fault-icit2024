% Copyright (c) 2024 Tuomas Jalonen & Mohammad Al-Sa'd
%
% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files (the "Software"),
% to deal in the Software without restriction, including without limitation
% the rights to use, copy, modify, merge, publish, distribute, sublicense,
% and/or sell copies of the Software, and to permit persons to whom the
% Software is furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
% THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
% FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
% DEALINGS IN THE SOFTWARE.
%
% Emails: tuomas.jalonen@tuni.fi & mohammad.al-sad@helsinki.fi
%
% The following reference should be cited whenever this script is used:
% Jalonen, T., Al-Sa'd, M., Kiranyaz, S., & Gabbouj, M. (2024). Real-Time
% Vibration-Based Bearing Fault Diagnosis Under Time-Varying Speed Conditions.
% In 25th IEEE International Conference on Industrial Technology.
%
% Last Modification: 7-March-2024
%
% Description:
% This script performs the Fisher-based spectral separability analysis
% and generates the results present in Fig. 7.

%% Initialization
clear; close all; clc;

%% Parameters
data_folder = 'Results\KAIST Processed\';        % The preprocessed results directory
nfg         = true;                              % Noise flag true or false
SNR         = 5;                                 % Signal to noise ratio in dB
M           = 2048;                              % Number of frequency samples
fs          = 20e3;                              % Sampling frequency
r           = 2;                                 % Sensor number {1,2}
class_name  = {'Normal','Outer','Inner','Ball'}; % Bearing class names

%% Initialization
if nfg == true
    folder = [data_folder num2str(SNR) ' dB\'];
else
    folder = [data_folder 'Clean\'];
end

%% Power Spectral Density
disp('Estimating the power spectral densities ...');
Ps = cell(1,length(class_name));
for p = 1:length(class_name)
    disp(class_name{p});
    folder_name = [folder class_name{p}];
    Q = (length(dir(folder_name))-2);
    Ps{p} = zeros(M/2, Q);
    for q = 1:Q
        load([folder_name '\sample_' num2str(q)]);
        temp = fft((s(:,r) - mean(s(:,r)))./std(s(:,r)),M);
        Ps{p}(:,q) = smoothdata(abs(temp(1:end/2)).^2/(size(s,1)-1), 'gaussian', M/128);
    end
end

%% Fisher multivariate criterion
disp('Apply the Fisher-based spectral separability analysis ...');
thresh = 2;
cnt = 0;
mask = zeros(length(class_name)*(length(class_name)-1)/2,M/2);
for i = 1:length(class_name)
    x1 = 10*log10(Ps{i});
    for j = (i+1):length(class_name)
        cnt = cnt + 1;
        x2 = 10*log10(Ps{j});
        mask(cnt,:) = ((mean(x1,2)-mean(x2,2)).^2)./(var(x1,[],2) + var(x2,[],2));
    end
end
mask(mask <= thresh) = 0;
for i = 1:size(mask,1)
    mask(i,:) = smoothdata(mask(i,:),'Gaussian',M/32);
end

%% Plotting
f = 0:fs/(M-1):fs/2; f = f./1000;
Color = [31, 119, 180; 255, 127, 14; 148, 103, 189; 44, 160, 44]./255;
alpha = 0.2;
yname = {'Normal - Outer','Normal - Inner','Normal - Ball',...
    'Outer - Inner' ,'Outer - Ball','Inner - Ball'};
figure('Color',[1,1,1],'Position',[25 50 1000 700]); 
t = tiledlayout(7,1,"TileSpacing","tight","Padding","tight");
nexttile(1,[2 1]);
imagesc(f,1:size(mask,1),mask,'AlphaData',0.7); colorbar; colormap(turbo);
yticks(1:(size(mask,1))); axis([0 f(end) 0.5 size(mask,1)+0.5]); grid on;
set(gca,'XTicklabel','','YTicklabel',yname,'fontweight','bold', ...
    'fontsize',22,'TickLabelInterpreter','latex'); clim([0 8]);
title('Inter-Class Separability','Interpreter','latex','FontSize',26); box on;

nexttile(3,[5 1]); p = [];
for i = length(class_name):-1:1
    x = 10*log10(Ps{i});
    xm = mean(x,2); xp = xm + std(x,[],2); xn = xm - std(x,[],2);
    p(i) = plot(f,xm,'linewidth',1.5,'Color', Color(i,:)); hold on;
    patch = fill([f fliplr(f)]', [xp' fliplr(xn')]', Color(i,:));
    set(patch,'edgecolor','none','FaceAlpha',alpha);
end
set(gca,'fontweight','bold','FontSize',18); grid on;
ylabel('Power Spectral Density (dB)','Interpreter','latex','FontSize',24);
xlabel('Frequency (kHz)','Interpreter','latex','FontSize',24);
legend(p,class_name,'Interpreter','latex','FontSize',20, ...
    'location','northeast','orientation','horizontal');
set(gcf,'Units','inches'); screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',screenposition(3:4));
