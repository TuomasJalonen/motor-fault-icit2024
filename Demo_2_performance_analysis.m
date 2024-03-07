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
% This script generates the performance analysis results in Table 1 and Figs. 3 and 4.

%% Initialization
clear; close all; clc;

%% Parameters
SNR = [25 20:-5:-5];
Labels = {'Clean' '20 dB' '15 dB' '10 dB' '5 dB' '0 dB' '-5 dB'};
yLabel = {'Accuracy (\%)' 'Precision (\%)' 'Recall (\%)' 'F1-Score (\%)'};
fname  = {'clean' '20dB' '15dB' '10dB' '5dB' '0dB' '-5dB'};

%% Our Results
cnn_avg_perf = zeros(length(yLabel),length(SNR));
cnn_std_perf = zeros(length(yLabel),length(SNR));
cnn_confuse  = zeros(length(SNR),4);
for i = 1:length(fname)
    fid = fopen(['Results\Performance\cross-validation_metrics_' fname{i} '.json']);
    str = char(fread(fid,inf)'); fclose(fid); val = jsondecode(str);
    cnn_avg_perf(:,i) = [val.avg_acc; val.avg_precisions; val.avg_recalls; val.avg_f1s];
    cnn_std_perf(:,i) = [val.std_acc; val.std_precisions; val.std_recalls; val.std_f1s];
end
cnn_confuse(1,:)  = [0.965 0.981 1.00 0.990];
cnn_confuse(2,:)  = [0.824 0.873 0.998 0.955];
cnn_confuse(3,:)  = [0.693 0.765 0.992 0.912];
cnn_confuse(4,:)  = [0.543 0.694 0.956 0.819];

%% PIResNet Results from:
% Ni, Q., Ji, J. C., Halkon, B., Feng, K., & Nandi, A. K. (2023). Physics-Informed
% Residual Network (PIResNet) for rolling element bearing fault diagnostics.
% Mechanical Systems and Signal Processing, 200, 110544.
SNR_mod = [25 5 0 -5];
res_avg_perf = zeros(4,4);
res_std_perf = zeros(4,4);
res_confuse  = zeros(4,4);
res_avg_perf(1,:) = [0.978 0.903 0.813 0.717];
res_avg_perf(2,:) = [0.980 0.905 0.816 0.723];
res_avg_perf(3,:) = [0.980 0.905 0.816 0.720];
res_avg_perf(4,:) = [0.980 0.905 0.816 0.722];
res_std_perf(1,:) = [0.002 0.002 0.004 0.002];
res_confuse(1,:)  = [0.965 0.968 0.998 0.989];
res_confuse(2,:)  = [0.860 0.860 0.997 0.903];
res_confuse(3,:)  = [0.740 0.756 0.987 0.782];
res_confuse(4,:)  = [0.637 0.664 0.920 0.661];

%% Plotting
Color = [31, 119, 180; 255, 127, 14]./255;
r = 2; alpha = 0.2;
for i = 1:length(yLabel)
    figure('Color',[1,1,1],'Position',[25 50 700 450]);
    p1 = plot(SNR_mod,100.*res_avg_perf(i,:),'LineWidth',2,'Color',Color(2,:), ...
        'Marker','sq','MarkerFaceColor',Color(2,:)); hold on;
    xp = res_avg_perf(i,:) + r.*res_std_perf(i,:);
    xn = res_avg_perf(i,:) - r.*res_std_perf(i,:);
    patch = fill([SNR_mod fliplr(SNR_mod)]', 100.*[xp fliplr(xn)]', Color(2,:));
    set(patch,'edgecolor','none','FaceAlpha',alpha);
    p2 = plot(SNR,100.*cnn_avg_perf(i,:),'LineWidth',2,'Color',Color(1,:), ...
        'Marker','o','MarkerFaceColor',Color(1,:)); grid on;
    xp = cnn_avg_perf(i,:) + r.*cnn_std_perf(i,:);
    xn = cnn_avg_perf(i,:) - r.*cnn_std_perf(i,:);
    patch = fill([SNR fliplr(SNR)]', 100.*[xp fliplr(xn)]', Color(1,:));
    set(patch,'edgecolor','none','FaceAlpha',alpha);
    axis([-7 27 70 101]);
    set(gca,'XDir','reverse','Xticklabels',fliplr(Labels), ...
        'fontweight','bold','FontSize',16,'TickLabelInterpreter','latex');
    ylabel(yLabel{i},'Interpreter','latex','FontSize',22);
    legend([p2, p1],'Proposed','PIResNet','Orientation','vertical', ...
    'Location','northeast','Interpreter','latex','FontSize',18);
    set(gcf,'Units','inches'); screenposition = get(gcf,'Position');
    set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',screenposition(3:4));
end