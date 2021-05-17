clc,clear
N_bs = [16 16];
N_BS = N_bs(1) * N_bs(2);
N_ms = 1;
fft_len = 256;
K = fft_len;
BW = 30.72e6;          %系统带宽 = 30.72MHz
fs = BW;
fc = 30e9;          % 子载波频率 is 30GHz
lambda = 3e8/fc;
d_ant = lambda/2;
sigma_2_alpha = 1;  % variance of path gain
sigma = 7.5/2;
tau_max = 0.2e-6;   % 绝对时延扩展 200ns 0.2μs
Nc = 6;
Np = 10;
A_R1 = dftmtx(N_bs(1))./sqrt(N_bs(1));
A_R2 = dftmtx(N_bs(2))./sqrt(N_bs(2));
P = kron(A_R1.', A_R2);
itermax = 500;
H_test = zeros(itermax*fft_len, 2, N_BS);       % 按照（样本数，实+虚）格式存储信道

%% generating channel
for i =1:itermax
    [H_fs_tmp] = channel_downlink(N_bs, N_ms, fc, Nc, Np, sigma_2_alpha,sigma,tau_max,fs,K); % (N_ms, N_BS, subcarrier)
    H_fs = permute(H_fs_tmp, [3,2,1]); % (subcarrier, N_BS, N_ms=1)
    H_fa = H_fs*P;
%     surf(abs(H_fa))
    H_tmp = zeros(K, 2, N_BS);
    for j = 1:K
        H = H_fa(j,:);
        H_re = real(H);
        H_im = imag(H);
        H_tmp(j,:,:) = [H_re;H_im]; 
    end
    H_test((i-1)*fft_len+1:i*fft_len, :, :) = H_tmp;

%     H_new_test(i,1,:,:) = real(H_channel);
%     H_new_test(i,2,:,:) = imag(H_channel);
    disp(['Finished','  ',num2str(i),'/',num2str(itermax)])
end 
% H_new_test = reshape(H_new_test, itermax, 2*fft_len*N_BS);
save H_test H_test
% save H_new_test H_new_test
disp('it is finished')


