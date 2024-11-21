%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��������
res = xlsread('���ݼ�.xlsx');

%%  ����ѵ�����Ͳ��Լ�
temp = randperm(719);

P_train = res(temp(1: 500), 1 : 28)';
T_train = res(temp(1: 500), 29: 31)';
M = size(P_train, 2);

P_test = res(temp(501: end), 1 : 28)';
T_test = res(temp(501: end), 29: 31)';
N = size(P_test, 2);

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  ����ƽ��
% ������ƽ�̳�1ά����ֻ��һ�ִ���ʽ
% Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
% ����Ӧ��ʼ�պ���������ݽṹ����һ��
p_train =  double(reshape(p_train, 7, 4, 1, M));%      7     4     1   500
p_test  =  double(reshape(p_test , 7, 4, 1, N));%      7     4     1   219
t_train =  double(t_train)';
t_test  =  double(t_test )';
disp(size(p_test));
%%  ��������ṹ
layers = [
 imageInputLayer([7, 4, 1])             % �������������Ϊ7 * 4
 
 convolution2dLayer([2, 1], 8)          % ��һ������� ����˴�СΪ2 * 1
 batchNormalizationLayer                % ����һ����
 reluLayer                              % relu�� �������
                                    
 convolution2dLayer([2, 2], 16)         % �ڶ�������� ����˴�СΪ 2 * 2
 batchNormalizationLayer                % ����һ����
 reluLayer                              % relu �����

 convolution2dLayer([2, 2], 32)         % �ڶ�������� ����˴�СΪ 2 * 2
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
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );
    
%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1', ps_output);
T_sim2 = mapminmax('reverse', t_sim2', ps_output);

%%  �����������ͼ
analyzeNetwork(layers)

for i = 1: 3

    %%  ���������
    error1(i, :) = sqrt(sum((T_sim1(i, :) - T_train(i, :)).^2) ./ M);
    error2(i, :) = sqrt(sum((T_sim2(i, :) - T_test (i, :)).^2) ./ N);
    
    %%  ��ͼ
    figure
    subplot(2, 1, 1)
    plot(1: M, T_train(i, :), 'r-*', 1: M, T_sim1(i, :), 'b-o', 'LineWidth', 1)
    legend('��ʵֵ', 'Ԥ��ֵ')
    xlabel('Ԥ������')
    ylabel('Ԥ����')
    string = {'ѵ����Ԥ�����Ա�'; ['RMSE=' num2str(error1(i, :))]};
    title(string)
    xlim([1, M])
    grid
    
    subplot(2, 1, 2)
    plot(1: N, T_test(i, :), 'r-*', 1: N, T_sim2(i, :), 'b-o', 'LineWidth', 1)
    legend('��ʵֵ', 'Ԥ��ֵ')
    xlabel('Ԥ������')
    ylabel('Ԥ����')
    string = {'���Լ�Ԥ�����Ա�'; ['RMSE=' num2str(error2(i, :))]};
    title(string)
    xlim([1, N])
    grid
    
    %%  �ָ���
    disp('**************************')
    disp(['���������', num2str(i)])
    disp('**************************')
    
    %%  ���ָ�����
    % ����ϵ�� R2
    R1(i, :) = 1 - norm(T_train(i, :) - T_sim1(i, :))^2 / norm(T_train(i, :) - mean(T_train(i, :)))^2;
    R2(i, :) = 1 - norm(T_test (i, :) - T_sim2(i, :))^2 / norm(T_test (i, :) - mean(T_test (i, :)))^2;
    
    disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1(i, :))])
    disp(['���Լ����ݵ�R2Ϊ��', num2str(R2(i, :))])
    
    % ƽ��������� MAE
    mae1(i, :) = sum(abs(T_sim1(i, :) - T_train(i, :))) ./ M ;
    mae2(i, :) = sum(abs(T_sim2(i, :) - T_test (i, :))) ./ N ;
    
    disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1(i, :))])
    disp(['���Լ����ݵ�MAEΪ��', num2str(mae2(i, :))])
    
    % ƽ�������� MBE
    mbe1(i, :) = sum(T_sim1(i, :) - T_train(i, :)) ./ M ;
    mbe2(i, :) = sum(T_sim2(i, :) - T_test (i, :)) ./ N ;
    
    disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1(i, :))])
    disp(['���Լ����ݵ�MBEΪ��', num2str(mbe2(i, :))])

end