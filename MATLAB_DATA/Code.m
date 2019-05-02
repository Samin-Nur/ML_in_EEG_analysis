load ('Input.mat')
Feature_set=[];
for i=1:1200
   EEG_seperated=Input(1+(i-1)*3000:3000+(i-1)*3000);

EEG_mean=mean(EEG_seperated);
EEG_seperated=EEG_seperated-mean(EEG_seperated); 
log_variance=log(var(EEG_seperated));
EEG_seperated=EEG_seperated/sqrt(var(EEG_seperated));

EEG_energy=sum(EEG_seperated.^2);
EEG_power=EEG_energy/length(EEG_seperated);
L=16384;
overall_spectrum=fft(EEG_seperated,L);
spectrum = overall_spectrum(1:L/2+1);
Power = (spectrum.*conj(spectrum));
Power(2:end-1)=2*Power(2:end-1);

samp_per_freq=length(Power)/50;
delta_energy=sum(Power(1:round(4*samp_per_freq)))/sum(Power(1:end));
theta_energy=sum(Power(round(4*samp_per_freq)+1:round(8*samp_per_freq)))/sum(Power(1:end));
alpha_energy=sum(Power(round(8*samp_per_freq)+1:round(12*samp_per_freq)))/sum(Power(1:end));
beta_energy=sum(Power(round(12*samp_per_freq)+1:round(35*samp_per_freq)))/sum(Power(1:end));
gamma_energy=sum(Power(round(35*samp_per_freq)+1:round(50*samp_per_freq)))/sum(Power(1:end));
average_power_delta=mean(Power(1:round(4*samp_per_freq)));
average_power_theta=mean(Power(round(4*samp_per_freq)+1:round(8*samp_per_freq)));
average_power_alpha=mean(Power(round(8*samp_per_freq)+1:round(12*samp_per_freq)));
average_power_beta=mean(Power(round(12*samp_per_freq)+1:round(35*samp_per_freq)));
average_power_gamma=mean(Power(round(35*samp_per_freq)+1:round(50*samp_per_freq)));
ratio1=average_power_delta/average_power_theta;
ratio2=average_power_theta/average_power_alpha;
ratio3=average_power_alpha/average_power_beta;
ratio4=average_power_beta/average_power_gamma;

%other features
delta_energy_window=0;
theta_energy_window=0;
alpha_energy_window=0;
beta_energy_window=0;
gamma_energy_window=0;
delta_power_window=0;
theta_power_window=0;
alpha_power_window=0;
beta_power_window=0;
gamma_power_window=0;

for n=1:1:(floor(length(EEG_seperated)/100))

    EEG_seperated_window=EEG_seperated((n-1)*100+1:(n-1)*100+100).';
    
    L=4096;
    overall_spectrum_window=fft(EEG_seperated_window,L);
    spectrum_window = overall_spectrum_window(1:L/2+1);
    power_window = (spectrum_window.*conj(spectrum_window));
    power_window(2:end-1)=2*power_window(2:end-1);
    
samp_per_freq_window=length(power_window)/50;
delta_energy_window(n)=sum(power_window(1:round(4*samp_per_freq_window)))/sum(power_window(1:end));
theta_energy_window(n)=sum(power_window(round(4*samp_per_freq_window)+1:round(8*samp_per_freq_window)))/sum(power_window(1:end));
alpha_energy_window(n)=sum(power_window(round(8*samp_per_freq_window)+1:round(12*samp_per_freq_window)))/sum(power_window(1:end));
beta_energy_window(n)=sum(power_window(round(12*samp_per_freq_window)+1:round(35*samp_per_freq_window)))/sum(power_window(1:end));
gamma_energy_window(n)=sum(power_window(round(35*samp_per_freq_window)+1:round(50*samp_per_freq_window)))/sum(power_window(1:end));
delta_power_window(n)=mean(power_window(1:round(4*samp_per_freq_window)));
theta_power_window(n)=mean(power_window(round(4*samp_per_freq_window)+1:round(8*samp_per_freq_window)));
alpha_power_window(n)=mean(power_window(round(8*samp_per_freq_window)+1:round(12*samp_per_freq_window)));
beta_power_window(n)=mean(power_window(round(12*samp_per_freq_window)+1:round(35*samp_per_freq_window)));
gamma_power_window(n)=mean(power_window(round(35*samp_per_freq_window)+1:round(50*samp_per_freq_window)));


end

ratio1_window=delta_power_window./theta_power_window;
ratio2_window=theta_power_window./alpha_power_window;
ratio3_window=alpha_power_window./beta_power_window;
ratio4_window=beta_power_window./gamma_power_window;
delta_energy_window_max=max(delta_energy_window);
delta_energy_window_min=min(delta_energy_window);
delta_energy_window_median=median(delta_energy_window);
theta_energy_window_max=max(theta_energy_window);
theta_energy_window_min=min(theta_energy_window);
theta_energy_window_median=median(theta_energy_window);
alpha_energy_window_max=max(alpha_energy_window);
alpha_energy_window_min=min(alpha_energy_window);
alpha_energy_window_median=median(alpha_energy_window);
beta_energy_window_max=max(beta_energy_window);
beta_energy_window_min=min(beta_energy_window);
beta_energy_window_median=median(beta_energy_window);
gamma_energy_window_max=max(gamma_energy_window);
gamma_energy_window_min=min(gamma_energy_window);
gamma_energy_window_median=median(gamma_energy_window);
ratio1_window_max=max(ratio1_window);
ratio1_window_min=min(ratio1_window);
ratio1_window_median=median(ratio1_window);
ratio2_window_max=max(ratio2_window);
ratio2_window_min=min(ratio2_window);
ratio2_window_median=median(ratio2_window);
ratio3_window_max=max(ratio3_window);
ratio3_window_min=min(ratio3_window);
ratio3_window_median=median(ratio3_window);
ratio4_window_max=max(ratio4_window);
ratio4_window_min=min(ratio4_window);
ratio4_window_median=median(ratio4_window);


% EEG_renyi=0;
% EEG_number_of_ridges=0;
% EEG_ridges_deltatheta=0;
% EEG_ridges_alphabeta=0;
% EEG_ridges_gamma=0;
% TF_window_length=500;
% 
% for n=1:1:(floor(length(EEG_seperated)/TF_window_length))
% 
%     EEG_seperated_window=EEG_seperated((n-1)*TF_window_length+1:(n-1)*TF_window_length+TF_window_length).';
%  h=hamming(21);
%  [tfr,rtfr,hat] = tfrrsp(EEG_seperated_window,1:length(EEG_seperated_window),512,h);
% 
% EEG_renyi(n)=renyi(tfr);
% ptt=0;
% ptf=0;
% [ptt,ptf]=ridges(tfr,hat);
% ptt=ptt/100;
% ptf=ptf*100;
% 
% deltatheta=0;
% alphabeta=0;
% gamma=0;
% 
% 
% for f=1:length(ptf)
%          if (ptf(f)<=8)
%              deltatheta=deltatheta+1;
%              else
%                  if (ptf(f)>8)&&(ptf(f)<=35)
%                  alphabeta=alphabeta+1;
%                      else
%                           if (ptf(f)>35)&&(ptf(f)<=50)
%                  gamma=gamma+1;
%                          end
%                  end
%          end
% end
%  
% 
% EEG_number_of_ridges(n)=deltatheta+alphabeta+gamma;
% EEG_ridges_deltatheta(n)=deltatheta;
% EEG_ridges_alphabeta(n)=alphabeta;
% EEG_ridges_gamma(n)=gamma;
% end
% 
% EEG_max_renyi=max(EEG_renyi);
% EEG_min_renyi=min(EEG_renyi);
% EEG_median_renyi=median(EEG_renyi);
% EEG_number_of_ridges_avg=mean(EEG_number_of_ridges);
% EEG_ridges_deltatheta_max=max(EEG_ridges_deltatheta);
% EEG_ridges_deltatheta_min=min(EEG_ridges_deltatheta);
% EEG_ridges_deltatheta_median=median(EEG_ridges_deltatheta);
% EEG_ridges_alphabeta_max=max(EEG_ridges_alphabeta);
% EEG_ridges_alphabeta_min=min(EEG_ridges_alphabeta);
% EEG_ridges_alphabeta_median=median(EEG_ridges_alphabeta);
% EEG_ridges_gamma_max=max(EEG_ridges_gamma);
% EEG_ridges_gamma_min=min(EEG_ridges_gamma);
% EEG_ridges_gamma_median=median(EEG_ridges_gamma);


TheMATRIX=[
EEG_mean;
log_variance;
EEG_power;
delta_energy;
theta_energy;
alpha_energy;
beta_energy;
gamma_energy;
ratio1;
ratio2;
ratio3;
ratio4;
delta_energy_window_max;
delta_energy_window_min;
delta_energy_window_median;
theta_energy_window_max;
theta_energy_window_min;
theta_energy_window_median;
alpha_energy_window_max;
alpha_energy_window_min;
alpha_energy_window_median;
beta_energy_window_max;
beta_energy_window_min;
beta_energy_window_median;
gamma_energy_window_max;
gamma_energy_window_min;
gamma_energy_window_median;
ratio1_window_max;
ratio1_window_min;
ratio1_window_median;
ratio2_window_max;
ratio2_window_min;
ratio2_window_median;
ratio3_window_max;
ratio3_window_min;
ratio3_window_median;
ratio4_window_max;
ratio4_window_min;
ratio4_window_median;

];

% EEG_max_renyi;
% EEG_min_renyi;
% EEG_median_renyi;
% EEG_number_of_ridges_avg;
% EEG_ridges_deltatheta_max;
% EEG_ridges_deltatheta_min;
% EEG_ridges_deltatheta_median;
% EEG_ridges_alphabeta_max;
% EEG_ridges_alphabeta_min;
% EEG_ridges_alphabeta_median;
% EEG_ridges_gamma_max;
% EEG_ridges_gamma_min;
% EEG_ridges_gamma_median
Feature_set=[Feature_set, TheMATRIX];
end
save('Feature_set.mat','Feature_set')