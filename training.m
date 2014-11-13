run('C:/Users/Valerie/Documents/MATLAB/vlfeat-0.9.19/toolbox/vl_setup.m');

% load images
load('possamples.mat'); npos=size(possamples,3);
load('negsamples.mat'); nneg=size(negsamples,3);  
possamples=double(possamples);
negsamples=double(negsamples);
fprintf('load  %6d positive samples from %s\n',npos,posfname);
fprintf('load  %6d negative samples from %s\n\n',nneg,negfname);

possamples=meanvarpatchnorm(possamples);
negsamples=meanvarpatchnorm(negsamples);

%%%%%%%%%%%%%%% SVM training and classification assumes each sample
%%%%%%%%%%%%%%% in the form of a vector. Let's flatten training images
%%%%%%%%%%%%%%% into one vector per sample
%%%%%%%%%%%%%%%
xsz=size(possamples,2);
ysz=size(possamples,1);
Xpos=transpose(reshape(possamples,ysz*xsz,npos));
Xneg=transpose(reshape(negsamples,ysz*xsz,nneg));


%%%%%%%%%%%%%%% make vectore with sample labels:
%%%%%%%%%%%%%%% +1 for positives
%%%%%%%%%%%%%%% -1 for negatives
%%%%%%%%%%%%%%%
ypos=ones(npos,1);
yneg=-ones(nneg,1);

%%%%%%%%%%%%%%% separate data into the training and validation set
%%%%%%%%%%%%%%% 
ntrainpos=1000;
ntrainneg=1000;
indpostrain=1:ntrainpos; indposval=indpostrain+ntrainpos;
indnegtrain=1:ntrainneg; indnegval=indnegtrain+ntrainneg;

Xtrain=[Xpos(indpostrain,:); Xneg(indnegtrain,:)];
ytrain=[ypos(indpostrain); yneg(indnegtrain)];
Xval=[Xpos(indposval,:); Xneg(indnegval,:)];
yval=[ypos(indposval); yneg(indnegval)];

% free memory
clear possamples negsamples

c = 0.001;       % c-parameter
epsilon = .000001;
kerneloption= 1; % degree of polynomial kernel (1=linear)
kernel='poly';   % polynomial kernel
verbose = 0;
tic
fprintf('Training SVM classifier with %d pos. and %d neg. samples...',sum(ytrain==1),sum(ytrain~=1))
[Xsup,yalpha,b,pos]=svmclass(Xtrain,ytrain,c,epsilon,kernel,kerneloption,verbose);
fprintf(' -> %d support vectors (%1.1fsec.)\n',size(Xsup,1),toc)


%%%%%%%%%%%%%%% get prediction for training and validation samples
%%%%%%%%%%%%%%%
fprintf('Running evaluation... ')
[ypredtrain,acctrain,conftrain]=svmvalmod(Xtrain,ytrain,Xsup,yalpha,b,kernel,kerneloption);
[ypredval,accval,confval]=svmvalmod(Xval,yval,Xsup,yalpha,b,kernel,kerneloption);
fprintf('Training accuracy: %1.3f; validation accuracy: %1.3f\n',acctrain(1),accval(1))
fprintf('press a key...'), pause, fprintf('\n')



W = (yalpha'*Xsup)';
% 1.2 
conftrainnew = Xtrain*W+b;
confvalnew   = Xval*W+b;

%%%%%%%%%%%%%%% Re-compute classification accuracy using true sample labels 'y'
%%%%%%%%%%%%%%% for training and validation samples

acctrainnew = mean((conftrainnew>0)*2-1==ytrain);
accvalnew   = mean((confvalnew>0)*2-1==yval);
fprintf('Training and validation accuracy re-computed from W,b: %1.3f; %1.3f\n',acctrainnew,accvalnew)
fprintf('press a key...'), pause, fprintf('\n')


%%%%%%%%%%%%%%% The values of the hyper-plane W are
%%%%%%%%%%%%%%% the weights of individual pixel values and can be displayed
%%%%%%%%%%%%%%% as an image. Let's construct W from support vectors
%%%%%%%%%%%%%%%
clf, showimage(reshape(W,24,24))

%%%%%%%%%%%%%%% the validation set. (Note that C or any other parameters
%%%%%%%%%%%%%%% should never be optimized on the final test set.)
%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% TODO:
%%%%%%%%%%%%%%% 2.1 train linear SVM as above for different values of C
%%%%%%%%%%%%%%%     and select the best C 'Cbest' and the best model 'Wbest'
%%%%%%%%%%%%%%%     and 'bbest' maximizing accuracy on the validation set.
%%%%%%%%%%%%%%%     Hint: try exponentially distribution values, e.g. 
%%%%%%%%%%%%%%%     [1000 100 10 1 .1 .01 .001 .0001 .00001];
%%%%%%%%%%%%%%% 2.2 When traibning for different C values, compute W,b
%%%%%%%%%%%%%%%     and visualize W as an image (see above)
%%%%%%%%%%%%%%%
Call=[1000 100 10 1 .1 .01 .001 .0001 .00001];
accbest=-inf; 
modelbest=[];
for i=1:length(Call)
  C=Call(i);
  
  % fill-in this part with the training of linear SVM for
  % the current C value (scode above). Select the model 
  % 'modelbest' maximizing accuracy on the validation set. 
  % Compute and display W for the current model
  
  [Xsup,yalpha,b,pos]=svmclass(Xtrain,ytrain,C,epsilon,kernel,kerneloption,verbose);
  [ypredtrain,acctrain,conftrain]=svmvalmod(Xtrain,ytrain,Xsup,yalpha,b,kernel,kerneloption);
  [ypredval,accval,confval]=svmvalmod(Xval,yval,Xsup,yalpha,b,kernel,kerneloption);
  W = (yalpha'*Xsup)';
  %clf, showimage(reshape(W,24,24));
  s=sprintf('C=%1.5f | Training accuracy: %1.3f; validation accuracy: %1.3f',C,acctrain,accval);
  title(s); fprintf([s '\n']); drawnow
  if accbest<accval,
      accbest = accval;
      Cbest = C;
      Wbest = W;
      bbest = b;
  end
end
fprintf(' -> Best accuracy %1.3f for C=%1.5f\n',accbest,Cbest)
fprintf('press a key...'), pause, fprintf('\n')

[x,y] = Stage1Detector( double(rgb2gray(img)), reshape(Wbest,24,24) );
%save('whatever.mat','Wbest', 
hog = vl_hog(single(img),8, 'verbose') ;
imhog = vl_hog('render', hog, 'verbose') ;
clf ; imagesc(imhog) ; colormap gray ;

