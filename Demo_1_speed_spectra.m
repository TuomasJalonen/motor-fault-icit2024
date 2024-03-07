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
% This script generates the frequency analysis results in Fig. 2.

%% Initialization
clear; close all; clc;

%% Parameters
data_folder = 'Dataset\';                        % The KAIST raw dataset directory
K           = 7;                                 % Number of subsets in the KAIST dataset
class_name  = {'normal','inner','outer','ball'}; % Bearing class names
NUFFT_N     = 2^12;                              % The number of NUFFT samples

%% Loading speed signals
disp('Loading speed signals ...');
x = cell(length(class_name),K);
t = cell(length(class_name),K);
T = inf;
for i = 1:length(class_name)
    for j = 1:K
        temp = readmatrix([data_folder 'rpm_' class_name{i} '_' num2str(j-1) '.csv']);
        T = min([min(diff(temp(:,1))) T]);
        x{i,j}(:,2) = [temp(:,2)];
        x{i,j}(:,1) = [temp(:,1)];
    end
end

%% Frequency analysis
disp('Frequency analysis ...');
fs = 1/T; cnt = 0;
f = (0:NUFFT_N-1)/NUFFT_N*fs;
xf  = zeros(length(class_name)*K,NUFFT_N);
for i = 1:length(class_name)
    for j = 1:K
        cnt = cnt + 1;
        temp = nufft(x{i,j}(:,2),x{i,j}(:,1)*fs);
        xf(cnt,:) = resample(temp,NUFFT_N,length(temp));
        xf(cnt,:) = abs(xf(cnt,:)).^2./(NUFFT_N-1);
    end
end
xf_norm = mean(xf)./sum(mean(xf));
xf_area = sum(xf_norm(f<=10));
disp([num2str(round(100.*xf_area,1)) '% of the total power spectrum']);
xf = 10*log10(xf);
xp = mean(xf) + std(xf);
xn = mean(xf) - std(xf);

%% Plotting
i = 2; j = 7;
figure('Color',[1,1,1],'Position',[25 50 950 250]);
plot(x{i,j}(:,1)./60,x{i,j}(:,2),'linewidth',1.5,'Color','k');
axis([0 5.02 500 2530]); grid on;
set(gca,'TickLabelInterpreter','latex','FontSize',16);
ylabel('Speed (RPM)','Interpreter','latex','FontSize',22);
xlabel('Time (minutes)','Interpreter','latex','FontSize',22);
set(gcf,'Units','inches'); screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',screenposition(3:4));

figure('Color',[1,1,1],'Position',[25 50 950 450]);
p2 = fill([f fliplr(f)]', [xp fliplr(xn)]', [0.7 0.7 0.7]);
set(p2,'edgecolor','none','FaceAlpha',1); hold on;
p1 = plot(f,(mean(xf)),'linewidth',1.5,'Color','k'); hold on;
p3 = plot(repelem(10,1,100),linspace(-200,200,100),'Color','r','LineWidth',2);
grid on; axis([0 f(end) 15 92]);
set(gca,'TickLabelInterpreter','latex','FontSize',16);
ylabel('Power (dB)','Interpreter','latex','FontSize',22);
legend([p1 p2 p3],'Averaged','$\mu\pm\sigma$','Threshold', 'Orientation','horizontal', ...
    'Location','northwest','Interpreter','latex','FontSize',18);
xlabel('Frequency (Hz)','Interpreter','latex','FontSize',22);
set(gcf,'Units','inches'); screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',screenposition(3:4));
