warning off;
close all;
clear all;
clc;
%% Load Input Signal

j=sqrt(-1);
[filename,pathname]=uigetfile('.wav','Pick the wav file');%%getting input of wav file(mp3)
[wavin,fs]=audioread([pathname filename]);
p = audioplayer(wavin, fs);
play(p);
wav_length=length(wavin);
figure,plot(wavin);

%% Noise Classsification

switch fs 
     case 8000
         frame_len=320;step_len=160;
         msgbox('babble noise')
     case 10000
         frame_len=400;step_len=200;
          msgbox('factory noise')
     case 12000
         frame_len=480;step_len=240;
          msgbox('pink noise')
     case 16000
         frame_len=640;step_len=320;
           msgbox('volvo noise') 
     case 44100
         frame_len=1800;step_len=900;
          msgbox('white noise')
     otherwise
         frame_len=1800;step_len=900;
        msgbox('babble noise')
end;


%%
n_frame=fix((wav_length-frame_len)/step_len)+1; 
for i=1:n_frame
    n1=(i-1)*step_len+1;
    n2=(i-1)*step_len+frame_len;
    S(i,: ) =wavin(n1:n2);
end


%%  Fast Fourier Transform 

win_ham=hamming(frame_len); 
noise_foward15frame=zeros(n_frame,frame_len);
for nifrm=1:n_frame
    nf=fft(win_ham.*(S(nifrm,1:frame_len)'),frame_len);
    noise_foward15frame(nifrm, : ) =noise_foward15frame(nifrm, :) +(abs(nf)');
end
figure,plot(win_ham);

%%   
am_noise=mean(noise_foward15frame(:,1:frame_len));
sum_timedomain=zeros(n_frame,frame_len);
voice_timedomain=zeros(n_frame,frame_len);

%% Hidden Markov model

for ifrm=1:n_frame  
    sum_timedomain(ifrm,1:frame_len)=sum_timedomain(ifrm,1:frame_len)+(win_ham').*S(ifrm,1:frame_len);
    sf=fft((win_ham').*S(ifrm,:) ,frame_len); 
    phase=angle(sf);
    am_signal=abs(sf); 
    am_voice=am_signal-am_noise;
    
    %%
    for b=1:frame_len
        if(am_voice(1,b)<0)
            am_voice(1,b)=0;
        end
    end
    voice=am_voice.*exp(j*phase);
    sif=real(ifft(voice,frame_len));
    voice_timedomain(ifrm, :) =voice_timedomain(ifrm,:) +sif;
end

%%
wavout=zeros(1,wav_length);
for d=1:n_frame
    m1=(d-1)*step_len+1;
    m2=(d-1)*step_len+frame_len;
    wavout(m1:m2)=wavout(m1:m2)+voice_timedomain(d,1:frame_len);
end
figure,plot(wavout);
sound(wavout);

%% Calculate SNR

begin_speech = 25400; end_speech = 26800; 
snr_before = mean( (begin_speech:end_speech) .^2) / mean((begin_speech:end_speech) .^2); 
db_snr_before = 10*log10( snr_before );   % same thing, but in dB

% calculation on data after noise reduction follows: 
residual_noise = (begin_speech:end_speech) - (begin_speech:end_speech); 
snr_after = mean( (begin_speech:end_speech) .^ 2)/mean( residual_noise .^ 2); 
db_snr_after = 10*log10( snr_after );



%% Displaying Results
figure(4);
subplot(2,1,1);
plot(wavin);grid on; 
title('Input Signal');
axis([1 wav_length -1 1]);
subplot(2,1,2);
plot(wavout);grid on;
axis([1 wav_length -1 1]); 
title('Ouput Signal');
%% demos for HMM (hidden Bernoulli model)
addpath('HBM\')
d = 3;
k = 2;
n = fs;
[x, model] = hbmRnd(d, k, n);
%%
z = hbmViterbi(x,model);
%%
[alpha,llh] = hbmFilter(x,model);
%%
[gamma,alpha,beta,c] = hbmSmoother(x,model);
%%
[model, llh] = hbmEm(x,k);

sscs = msf_ssc(wavout,8000,12);sscs=mean(sscs);
rcs = msf_rc(wavout,8000,12);rcs=mean(rcs);
lpcs = msf_powspec(wavout,8000,12);lpcs=mean(mean(lpcs));
mfccs = msf_mfcc(wavout,8000,12);mfccs=mean(mfccs);
lsfs = msf_lsf(wavout,8000,12);lsfs=mean(lsfs);
lpccs = msf_lpcc(wavout,8000,12);lpccs=mean(lpccs);
% lpcs = msf_lpc(wavout,8000,12);lpcs=mean(mean(lpcs));
logfbs = msf_logfb(wavout,8000,12);logfbs=mean(logfbs);
lars = msf_lar(wavout,8000,12);lars=mean(lars);
lpcs = msf_filterbank(26,8000,0,8000,512);lpcs=mean(lpcs);
llh=mean(llh);
Testfea=[sscs rcs mfccs lsfs lpccs lpcs logfbs lars lpcs llh];
%% Classification

load label 
load Trainfea

svm=fitcecoc(Trainfea,label);
result=predict(svm,Testfea);

if result==1
    helpdlg('Female Angry');
elseif result==2
    helpdlg('Female Normal');
elseif result==3
    helpdlg('Female Surprise');
elseif result==4
    helpdlg('Male Angry');
else
    helpdlg('Male Normal');
end

            
    
