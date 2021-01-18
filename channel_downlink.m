function [H_fs] = channel_downlink(N_bs, N_ms, fc, Nc, Np, sigma_2_alpha, sigma, tau_max, fs, K)
%Inputs:
%   N_bs：基站天线数
%   N_ms：单天线用户
%   fc： 子载波频率
%   Nc： 簇
%   Np： 每一簇中路径数
%   sigma_2_alpha: 平均路径增益, generally, sigma_2_alpha = 1;
%   sigma： 角度扩展
%   tau_max： 最大延时
%   fs： 采样率
%   K： 子载波数量
%Outputs:
%   H_f：信道矩阵 空频域 N_ms*N_bs*K

%% 设置参数
lambda = 3e8/fc;
d_ant = lambda/2;
Lp = Nc*Np;
angle = sigma * pi / 180; % 角度扩展
ang_min = -pi/3 + 2*angle;
ang_max = pi/3 - 2*angle;
k = 2 * pi * d_ant / lambda;
%% 生成导引矢量
N_BS = N_bs(1) * N_bs(2);
N_MS = N_ms;

n_t = (0 : N_bs(1) - 1)';
m_t = (0 : N_bs(2) - 1)';

phi_BS = ang_min + (ang_max-ang_min)*rand(Nc,1); % 生成簇主径上的角度
phi_t = phi_BS * ones(1,Np) + ones(Nc,1) * ((rand(1,Np) - 0.5)*2*angle); % 生成簇上路径的角度
phi1 = reshape(phi_t.', [1,Lp]);
theta_BS = ang_min + (ang_max-ang_min)*rand(Nc,1);
theta_t = theta_BS*ones(1,Np) +ones(Nc,1)* ((rand(1,Np)-0.5)*2*angle); % 生成簇上路径的角度
theta1 = reshape(theta_t.', [1, Lp]);

A_t = zeros(N_BS, Lp);
for path = 1:Lp
    e_a1 = exp(-1i * k * sin(phi1(1, path)) * cos(theta1(1, path)) * n_t); % channel model 2
    e_e1 = exp(-1i * k * sin(theta1(1, path)) * m_t);
    A_t(:, path) = kron(e_a1, e_e1) / sqrt(N_BS);
end

n_r = (0:N_MS -1).';
phi_MS = ang_min + (ang_max-ang_min)*rand(Nc,1); % 生成簇主径上的角度
phi_r = phi_MS * ones(1,Np) + ones(Nc,1) * ((rand(1,Np) - 0.5)*2*angle); % 生成簇上路径的角度
phi2 = reshape(phi_r.', [1, Lp]);
A_r = zeros(N_MS, Lp);
for path = 1:Lp
    A_r(:,path) = exp(-1i * k * sin(phi2(1, path)) * n_r);
end


%% 生成时延 τ 路径增益 α
tau = tau_max*rand(1,Lp);
tau = sort(tau);
miu_tau = 2*pi*tau*fs/K;
alpha_temp = sqrt(sigma_2_alpha/2)*(randn(1,Lp) + 1i*randn(1, Lp));
alpha = sort(alpha_temp, 'descend');

H_fs = zeros(N_MS, N_BS, K);
for i = 1:K
    D_diag = sqrt(N_MS*N_BS/Lp)*diag(alpha.*exp(1i*(i-1)*miu_tau));
    
    H_fs(:,:,i) = A_r * D_diag * A_t';
    
end