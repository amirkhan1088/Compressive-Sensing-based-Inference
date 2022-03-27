%% For AT&T dataset
% link for datasets are given in description
categories = {'s1','s2','s3','s4','s5','s6','s7','s8','s9','s10',...
    's11','s12','s13','s14','s15','s16','s17','s18','s19','s20',...
    's21','s22','s23','s24','s25','s26','s27','s28','s29','s30',...
    's31','s32','s33','s34','s35','s36','s37','s38','s39','s40'};

rootFolder = 'AT&T';

%% For Extended Yale B Uncropped
rootFolder = 'ExtendedYaleB';

categories = {'yaleB11','yaleB12', 'yaleB13', 'yaleB15','yaleB16',...
    'yaleB17','yaleB18', 'yaleB19', 'yaleB20','yaleB21', 'yaleB22', 'yaleB23',...
    'yaleB24','yaleB25','yaleB26','yaleB27', 'yaleB28', 'yaleB29','yaleB30',...
    'yaleB31','yaleB32', 'yaleB33', 'yaleB34', 'yaleB35','yaleB36',...
    'yaleB37','yaleB38', 'yaleB39'};

%% For cropped Extended Yale B
rootFolder = 'ExtendedYaleB_Cropped'; % without ambient image files

categories = {'yaleB01','yaleB02', 'yaleB03','yaleB04', 'yaleB05','yaleB06',...
    'yaleB07','yaleB08', 'yaleB09', 'yaleB10','yaleB11','yaleB12', 'yaleB13', 'yaleB15','yaleB16',...
    'yaleB17','yaleB18', 'yaleB19', 'yaleB20','yaleB21', 'yaleB22', 'yaleB23',...
    'yaleB24','yaleB25','yaleB26','yaleB27', 'yaleB28', 'yaleB29','yaleB30',...
    'yaleB31','yaleB32', 'yaleB33', 'yaleB34', 'yaleB35','yaleB36',...
    'yaleB37','yaleB38', 'yaleB39'};
%% For GIT face datasets. This is the color dataset and hence need to convert to grey dataset
categories = {'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10',...
    's11','s12','s13','s14','s15','s16','s17','s18','s19','s20',...
    's21','s22','s23','s24','s25','s26','s27','s28','s29','s30',...
    's31','s32','s33','s34','s35','s36','s37','s38','s39','s40',...
    's41','s42','s43','s44','s45','s46','s47','s48','s49','s50'};

rootFolder = 'GIT';

%% Read the dataset and store the images in an imagestore
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource',...
    'foldernames');
%% Create training and test set by randomly selecting the samples from all groups and shuffle the sets
% two separate imagestores
[TrainFace TestFace] = splitEachLabel(imds,0.7,'randomized'); % 70% in training 30% in test set
%Shuffle
TrainFace = shuffle(TrainFace);
TestFace = shuffle(TestFace);
% labels
Y = TrainFace.Labels; % Labels for training set
Y1 = TestFace.Labels; % labels for test set

L = length(Y); % Number of samples in training set
L1 = length(Y1); % number of samples in test set

N = size(readimage(TrainFace,1),1)* size(readimage(TrainFace,1),2) % Total number of pixels in grey image
%% further processing
% C(j) is the fraction of pixels to be zeroed 
% Phi is the random measurement matrix. It is an MxN matrix where M is
% number of compressed samples and N is the total number of pixls in grey image
% here in this work Phi has been realized using mul and zero_indices
% variables

nr = size(readimage(TrainFace,1),1); %Number of rows in image
nc = size(readimage(TrainFace,1),2); %Number of columns in image

Xs = zeros(L,nc); % matrix containing the training compressed samples
XsT = zeros(L1,nc); %matrix containing the test compressed samples
C = [0.5 0.75 0.80 0.9 0.95 0.98 0.99];

Lc = length(C);
Accuracy = zeros(length(C),1); % Accuracy calculated for different percentage of zero pixels

zero_indices = zeros(nr,nc); % this hold the indices of pixels to be selected randomly and zeroed

% following loop generates the random column-wise indices to be made zero for each row
% these indices are same for whole dataset (training and test set) 
for i=1:nr
    zero_indices(i,:) = randperm(nc);
end

mu1 = randsrc(nr,nc); % this is the random +1 or -1 multiplier to each pixel

for j=1:Lc
    c = C(j);
    
    for i=1:L
        
        img = readimage(TrainFace,i);
        if ndims(img)>2
            img = rgb2gray(img); % convert the images grey if not already
        end

        img1 = double(img); %because randsrc results non uint8 integer 1 and -1
        img1 = img1.*mu1; % multiply each pixel with +-1
        for k=1:nr
            sel=zero_indices(k,1:ceil(c*nc)); % sel holds the indices of the pixels of each row to be made to zero
            img1(k,sel)= 0; % selected pixels are made to zeros
        end
        
        m = sigma_delta_UD_Counter_col(img1,255); % perform the sigma-delta 
        %quantization followed by decimation using an UP-DOWN counter
        % you can select the reference for sigma-delta as 0 -255 or 127.5 or 255 
        
        %m = mean(img1); % sigma-delta performs averaging and qunatization
        %which is similar to mean function but this function has higher
        %precision which may not be available in hardware
        
        Xs(i,:) = m; % compressed samples are stored 

    end
 % all the above steps repeated for the test set
    for i=1:L1
        img = readimage(TestFace,i);
        if ndims(img)>2
            img = rgb2gray(img);
        end

        img1 = double(img);
        img1 = img1.*mu1; % multiply each pixel with +-1
        for k=1:nr
            sel=zero_indices(k,1:ceil(c*nc));
            img1(k,sel)= 0; % selected pixels are made to zeros
        end
        %m = mean(img1);
        m = sigma_delta_UD_Counter_col(img1,255);
       

        XsT(i,:) = m;
    end

% multi-class support vector machine fitcecoc is used
    t = templateSVM('KernelFunction','Linear');
    mdl = fitcecoc(Xs,Y,'Coding','onevsall','Learners',t);
    pred = predict(mdl,XsT);
    acc = sum(pred==Y1)/L1
    Accuracy(j) = acc;
    % following two lines can be uncommented can be uncommented if 
    %cross validation is to be performed
    
%     CVMDL = crossval(mdl);
%     genError = kfoldLoss(CVMDL)
end