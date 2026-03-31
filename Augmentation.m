clc; clear; close all;

inputFolder  = 'C:\';
labelsInputFolder = 'C:\';
outputFolder = 'C:\';
labelsFolder = fullfile(outputFolder, 'labels');

if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end
if ~exist(labelsFolder, 'dir'), mkdir(labelsFolder); end

brightnessRange = [-0.40, +0.23];
blurSigmaMax    = 10;
noiseLevel      = 0.080;

imgFiles = dir(fullfile(inputFolder, '*.jpg'));
fprintf('Przetwarzam zbiór TRAIN: %d obrazów\n', numel(imgFiles));

for k = 1:numel(imgFiles)
    fname = imgFiles(k).name;
    imgPath = fullfile(inputFolder, fname);
    img = imread(imgPath);

    [~, name, ext] = fileparts(fname);

    txtFile = fullfile(labelsInputFolder, [name '.txt']);
    if exist(txtFile, 'file')
        labels = readmatrix(txtFile);
    else
        labels = [];
    end

    saveLabels = @(lbls, newName) inlineSaveLabels(lbls, newName, labelsFolder);

    newName = sprintf('%s_orig', name);
    imwrite(img, fullfile(outputFolder, [newName ext]));
    saveLabels(labels, newName);

    brightFactor = 1 + (brightnessRange(1) + (brightnessRange(2)-brightnessRange(1))*rand());
    newName = sprintf('%s_bright%.2f', name, brightFactor);
    imgBright = im2double(img) * brightFactor;
    imgBright = im2uint8(max(min(imgBright,1),0));
    imwrite(imgBright, fullfile(outputFolder, [newName ext]));
    saveLabels(labels, newName);

    sigma = blurSigmaMax;
    newName = sprintf('%s_blur%.2f', name, sigma);
    imgBlur = imgaussfilt(img, sigma);
    imwrite(imgBlur, fullfile(outputFolder, [newName ext]));
    saveLabels(labels, newName);

    newName = sprintf('%s_noise%.3f', name, noiseLevel);
    imgNoisy = imnoise(img, 'gaussian', 0, noiseLevel);
    imwrite(imgNoisy, fullfile(outputFolder, [newName ext]));
    saveLabels(labels, newName);

    imgD = im2double(img);
    effects = {@addSnow, @addFog, @addRain, @addNight};
    effectNames = {'snow','fog','rain','night'};

    nCombinations = 3;
    for c = 1:nCombinations
        imgAug = imgD;
        comboName = '';
        for e = 1:length(effects)
            if rand() > 0.5
                switch effectNames{e}
                    case 'snow', imgAug = addSnow(imgAug, 0.04);
                    case 'fog',  imgAug = addFog(imgAug, 0.5);
                    case 'rain', imgAug = addRain(imgAug, 0.02);
                    case 'night',imgAug = addNight(imgAug);
                end
                comboName = [comboName '_' effectNames{e}];
            end
        end

        if isempty(comboName), comboName = '_none'; end

        newName = sprintf('%s%s', name, comboName);

        imwrite(imgAug, fullfile(outputFolder, [newName ext]));
        saveLabels(labels, newName);
    end
end

fprintf('Augmentacja zakończona!\n');

function inlineSaveLabels(labels, newName, labelsFolder)
    newTxtFile = fullfile(labelsFolder, [newName '.txt']);
    fileID = fopen(newTxtFile,'w');
    if fileID == -1
        error('Nie można utworzyć pliku: %s', newTxtFile);
    end
    if ~isempty(labels)
        for i = 1:size(labels,1)
            fprintf(fileID,'%d %.6f %.6f %.6f %.6f\n', ...
                labels(i,1), labels(i,2), labels(i,3), labels(i,4), labels(i,5));
        end
    end
    fclose(fileID);
end

function out = addSnow(img, intensity)
    [h, w, ~] = size(img);
    snow = rand(h, w) > (1 - 2*intensity);
    motion = fspecial('motion', 20, 45);
    snowBlur = imfilter(double(snow), motion, 'replicate');
    out = img + repmat(snowBlur,1,1,3);
    out = min(out,1);
end

function out = addFog(img, strength)
    [h, w, ~] = size(img);
    h4 = max(floor(h/4),1);
    w4 = max(floor(w/4),1);
    fog = imresize(imgaussfilt(rand(h4,w4),8), [h, w]);
    fog = mat2gray(fog) * 2*strength;
    out = img.*(1-fog) + fog;
end

function out = addRain(img, density)
    [h, w, ~] = size(img);
    drops = rand(h, w) > (1-density);
    motion = fspecial('motion', 25, 80);
    rain = imfilter(double(drops), motion, 'replicate');
    rain = mat2gray(rain) * 2;
    out = img - repmat(rain,1,1,3);
    out = max(out,0);
end

function out = addNight(img)
    darkened = img.^2.5;
    tint = cat(3,0.85,0.9,1.1);
    tinted = darkened .* tint;
    out = tinted * 0.75;
    out = min(max(out,0),1);
end
