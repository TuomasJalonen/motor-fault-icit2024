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
% This script generates the computational complexity analysis results in Fig. 6.

%% Initialization
clear; close all; clc;

%% Parameters
N = 10000;         % Number of samples for estimating the density function
t = 1000*0.022;    % Outliers threshold

%% Main
x = load('Results\Computational Complexity\monte_carlo_times.csv');
x = 1000*x;
y = x; y(y >= t) = [];
[f,xi] = ksdensity(y,linspace(0,25,N),'NumPoints',N,'Support','positive','Bandwidth',0.005);
u = mean(y);
s = std(y);

%% Plotting
figure('Color',[1,1,1],'Position',[25 50 750 350]);
a = histogram(x,N,"Normalization","pdf",'EdgeAlpha',0.1); hold on;
b = plot(xi,f,'LineWidth',3); grid on;
set(gca,'fontweight','bold','FontSize',14,'YtickLabels','');
axis([18.5 24.5 0 1.7]);
legend('Histogram',['Density ($' num2str(round(mean(y),1)) ...
    '\pm' num2str(round(std(y),2)) '$)'],'Interpreter','latex','FontSize',16);
xlabel('Time (milliseconds)','Interpreter','latex','FontSize',20);
set(gcf,'Units','inches'); screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',screenposition(3:4));
