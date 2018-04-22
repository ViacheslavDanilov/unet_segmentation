%% Clear workspace and command window 
clear all; close all; clc;
addpath(genpath(pwd));

%% Initialization
ext = '.png';
folderName = uigetdir(pwd);
listing = dir(fullfile(folderName,'*.mat'));
imagesFolderName = fullfile(folderName, 'images');
masksFolderName = fullfile(folderName, 'masks');
mkdir(imagesFolderName);
mkdir(masksFolderName);
numFiles = length(listing);
f = waitbar(0,'1','Name','Convertation',...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
setappdata(f,'canceling',0);

%% Convertation
for k = 1:numFiles
    % Update waitbar and message
    waitbar(k/numFiles,f,sprintf('Processed files: %d (%.f%%)', k, k*100/numFiles))
    % Check for clicked Cancel button
    if getappdata(f,'canceling')
        break
    end
    baseFileName = listing(k).name;
    fullFileName = fullfile(folderName, baseFileName);
    matFile = importdata(fullFileName);
    if matFile.label == 1
        label = 'meningioma';
    elseif  matFile.label == 2
        label = 'glioma';
    elseif matFile.label == 3
        label = 'pituitary_tumor';
    end
    baseFileName = erase(baseFileName, '.mat');
    baseFileName = str2double(baseFileName);
    baseFileName = num2str(baseFileName,'%04.f');  
    image = matFile.image;
    image = uint16(image);
    image = imadjust(image);
%     imageName = fullfile(imagesFolderName, ...
%                 strcat(baseFileName, '_' , label, '_image', ext)); % Default
    imageName = fullfile(imagesFolderName, ...
                strcat(baseFileName, '_tumor',  ext)); % U-net
    mask = matFile.tumorMask;
%     maskName = fullfile(masksFolderName, ...
%                strcat(baseFileName, '_' , label, '_mask', ext)); % Default
    maskName = fullfile(masksFolderName, ...
               strcat(baseFileName, '_tumor_mask', ext)); % U-net
    imwrite(image, imageName); 
    imwrite(mask, maskName);
end
delete(f)