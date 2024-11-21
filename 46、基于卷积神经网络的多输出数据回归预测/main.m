%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������


%%  ��������
res = xlsread('���ݼ�.xlsx');%    719    31

%%  ����ѵ�����Ͳ��Լ�
temp = randperm(719);

P_train = res(temp(1: 500), 1 : 28)';%     28   500
T_train = res(temp(1: 500), 29: 31)';%      3   500
M = size(P_train, 2); % 500

P_test = res(temp(501: end), 1 : 28)';%     28   219
T_test = res(temp(501: end), 29: 31)';%      3   219
N = size(P_test, 2);% 219

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  ����ƽ��
% ������ƽ�̳�1ά����ֻ��һ�ִ���ʽ
% Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
% ����Ӧ��ʼ�պ���������ݽṹ����һ��
p_train =  double(reshape(p_train, 28, 1, 1, M));%     28     1     1   500
p_test  =  double(reshape(p_test , 28, 1, 1, N));%     28     1     1   219
t_train =  double(t_train)';%    500     3
t_test  =  double(t_test )';%    219     3

%%  ��������ṹ
layers = [
 imageInputLayer([28, 1, 1])            % �������������Ϊ28 * 1
 
 convolution2dLayer([2, 1], 8)          % ��һ������� ����˴�СΪ2 * 1
 batchNormalizationLayer                % ����һ����
 reluLayer                              % relu�� �������
 
 maxPooling2dLayer([2, 1], 'Stride', 2) % ���ػ��� �ػ���СΪ2 * 1 ����Ϊ 2
                                    
 convolution2dLayer([2, 1], 16)         % �ڶ�������� ����˴�СΪ 2 * 1 
 batchNormalizationLayer                % ����һ����
 reluLayer                              % relu �����
 
 maxPooling2dLayer([2, 1], 'Stride', 2) % ���ػ��� �ػ���СΪ2 * 1 ����Ϊ 2

 convolution2dLayer([2, 1], 32)         % �ڶ�������� ����˴�СΪ 2 * 1 
 batchNormalizationLayer                % ����һ����
 reluLayer                              % relu �����

 fullyConnectedLayer(128)               % ȫ���Ӳ� ��Ԫ����Ϊ128��
 reluLayer                              % relu �����
 
 fullyConnectedLayer(3)                 % �����
 regressionLayer];

%%  ��������
options = trainingOptions('sgdm', ...   % �ݶȼ��㷽��ΪAdam
    'MiniBatchSize', 64, ...            % ��ѵ�� ÿ��ѵ��64������
    'MaxEpochs', 20, ...               % ���ѵ������Ϊ200��
    'InitialLearnRate', 0.001, ...      % ��ʼѧϰ��Ϊ0.001
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...     % ������ѧϰ��Ϊ0.001 * 0.1
    'LearnRateDropPeriod', 150, ...     % ѵ��150�κ� ѧϰ�ʽ��е���
    'Shuffle', 'every-epoch', ...       % ÿ��ѵ������˳��
    'ValidationPatience', Inf, ...      % �ر���֤
    'Plots', 'training-progress', ...   % ����ѵ������
    'ExecutionEnvironment', 'cpu', ...  % ����CPU����
    'Verbose', false);                  % �ر���������ʾ
%%  ѵ��ģ��
net = trainNetwork(p_train, t_train, layers, options);

%%  ģ��Ԥ��
%t_sim1 = predict(net, p_train);
%t_sim2 = predict(net, p_test );


%%  ���ݷ���һ��
% T_sim1 = mapminmax('reverse', t_sim1', ps_output);
% T_sim2 = mapminmax('reverse', t_sim2', ps_output);

%%  �����������ͼ
analyzeNetwork(layers)
