function SVM_of_example_classifier

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

% train and test classifier for each class
for i=1:VOCopts.nclasses
    cls=VOCopts.classes{i};
    classifier=train(VOCopts,cls);                  % train classifier
    test(VOCopts,cls,classifier);                   % test classifier
    [fp,tp,auc]=VOCroc(VOCopts,'comp1',cls,true);   % compute and display ROC
    
    if i<VOCopts.nclasses
        
        fprintf('press any key to continue with next class...\n');
        pause;
    end
end

% train classifier
function classifier = train(VOCopts,cls)

% load 'train' image set for class
[ids,classifier.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
disp(cls);
% extract features for each image
classifier.FD=zeros(0,length(ids));
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: train: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end

    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        fd=extractfd(VOCopts,I); %we extract here the feature vector fd= feature vector
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end
    %disp(fd);
    classifier.FD(1:length(fd),i)=fd;
end

% run classifier on test images
function test(VOCopts,cls,classifier)

% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d'); %carga les imatges per testejar

% create results file
fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');

% classify each image
tic;

classifier.mdl = fitcknn(classifier.FD.',classifier.gt,'NumNeighbors',2,'Standardize',1);
%classifier.mdl =fitcsvm(classifier.FD.',classifier.gt); %TRAIN THE CLASSIFIER

for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        fd=extractfd(VOCopts,I);
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end

    % compute confidence of positive classification
    c=classify_(VOCopts,classifier,fd);
    
    % write to results file
    fprintf(fid,'%s %f\n',ids{i},c);
end

% close results file

fclose(fid);

% trivial feature extractor: compute mean RGB
function fd = extractfd(VOCopts,I)
    net =resnet18;
    inputSize = net.Layers(1).InputSize;
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),I);
    layer = 'fc7'; 
    %fd = activations(net,augimdsTrain,layer,'OutputAs','rows')
    sz = net.Layers(1).InputSize;
    I = imresize(I,sz(1:2));
    [label,fd]= classify(net,I);
    disp(fd);
%fd=sque    eze(sum(sum(double(I)))/(size(I,1)*size(I,2)));

% trivial classifier: compute ratio of L2 distance betweeen
% nearest positive (class) feature vector and nearest negative (non-class)
% feature vector
function c = classify_(VOCopts,classifier,fd)
 [label,c,cost]=predict(classifier.mdl,fd);
 disp(c);
 c =c(2);




