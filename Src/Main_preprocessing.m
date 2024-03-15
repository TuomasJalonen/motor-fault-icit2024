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
% It pre-processes the vibration signals in the KAIST dataset.

%% Initialization
clear; close all; clc; rng(1);

%% Parameters
data_folder = 'Dataset\';                        % The KAIST raw dataset directory
out_folder  = 'Results\KAIST Processed\';        % The preprocessed results directory
nfg         = false;                             % Noise flag true or false
N           = 2000;                              % Numbe of samples in a segment
Fs          = 20e3;                              % New sampling frequency in Hz
SNR         = -5:5:20;                           % Signal to noise ratio in dB
K           = 7;                                 % Number of subsets in the KAIST dataset
vfs         = 25.6e3;                            % Vibration sampling frequency in Hz
t           = 0:1/Fs:300-1/Fs;                   % Time array for one subset
class_name  = {'normal','inner','outer','ball'}; % Bearing class names

%% Loading and resampling raw data
disp('Loading and resampling raw data ...')
x = cell(1,length(class_name));
for i = 1:length(class_name) % iterate across all classes
    disp(class_name{i})
    x{i} = [];
    for j = 1:K % iterate across all subsets
        temp = readmatrix([data_folder 'vibration_' class_name{i} '_' num2str(j-1) '.csv']);
        temp = resample(temp(:,3:4),Fs,vfs);
        x{i} = cat(1,x{i},temp);
    end
end

%% Segmenting and saving processed data
disp('Segmenting and saving processed data ...')
if nfg == true % when noise is needed
    for i = 1:length(class_name) % iterate across all classes
        disp(class_name{i})
        for k = 1:length(SNR) % iterate across all SNR levels
            disp(SNR(k)); dump = [];
            folder_name = [out_folder num2str(SNR(k)) ' dB\'...
                upper(class_name{i}(1)) class_name{i}(2:end)];
            n = randn(size(x{i}));
            n = (1./sqrt(mean(n.^2))).*n;
            n = sqrt(mean(x{i}.^2).*(10^(-1*SNR(k)/10))).*n;
            z = n + x{i};
            for j = 1:size(z,2) % iterate across all sensors
                dump = cat(3,dump,buffer(z(:,j), N, 0, 'nodelay')');
            end
            for j = 1:size(dump,1)
                s = squeeze(dump(j,:,:));
                save([folder_name '\sample_' num2str(j)],'s');
            end
        end
    end
else % when noise is not needed
    for i = 1:length(class_name) % iterate across all classes
        disp(class_name{i}); dump = [];
        folder_name = [out_folder 'Clean\'...
            upper(class_name{i}(1)) class_name{i}(2:end)];
        for j = 1:size(x{i},2) % iterate across all sensors
            dump = cat(3,dump,buffer(x{i}(:,j), N, 0, 'nodelay')');
        end
        for j = 1:size(dump,1)
            disp(100*j/size(dump,1));
            s = squeeze(dump(j,:,:));
            save([folder_name '\sample_' num2str(j)],'s');
        end
    end
end