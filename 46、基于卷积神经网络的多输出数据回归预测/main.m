%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行


%%  导入数据
res = xlsread('数据集.xlsx');%    719    31

%%  划分训练集和测试集
temp = randperm(719);

P_train = res(temp(1: 500), 1 : 28)';%     28   500
T_train = res(temp(1: 500), 29: 31)';%      3   500
M = size(P_train, 2); % 500

P_test = res(temp(501: end), 1 : 28)';%     28   219
T_test = res(temp(501: end), 29: 31)';%      3   219
N = size(P_test, 2);% 219

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  数据平铺
% 将数据平铺成1维数据只是一种处理方式
% 也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
% 但是应该始终和输入层数据结构保持一致
p_train =  double(reshape(p_train, 28, 1, 1, M));%     28     1     1   500
p_test  =  double(reshape(p_test , 28, 1, 1, N));%     28     1     1   219
t_train =  double(t_train)';%    500     3
t_test  =  double(t_test )';%    219     3

%%  构造网络结构
layers = [
 imageInputLayer([28, 1, 1])            % 输入层输入数据为28 * 1
 
 convolution2dLayer([2, 1], 8)          % 第一个卷积层 卷积核大小为2 * 1
 batchNormalizationLayer                % 批归一化层
 reluLayer                              % relu层 激活函数成
 
 maxPooling2dLayer([2, 1], 'Stride', 2) % 最大池化层 池化大小为2 * 1 步长为 2
                                    
 convolution2dLayer([2, 1], 16)         % 第二个卷积层 卷积核大小为 2 * 1 
 batchNormalizationLayer                % 批归一化层
 reluLayer                              % relu 激活层
 
 maxPooling2dLayer([2, 1], 'Stride', 2) % 最大池化层 池化大小为2 * 1 步长为 2

 convolution2dLayer([2, 1], 32)         % 第二个卷积层 卷积核大小为 2 * 1 
 batchNormalizationLayer                % 批归一化层
 reluLayer                              % relu 激活层

 fullyConnectedLayer(128)               % 全连接层 神经元个数为128个
 reluLayer                              % relu 激活层
 
 fullyConnectedLayer(3)                 % 输出层
 regressionLayer];

%%  参数设置
options = trainingOptions('sgdm', ...   % 梯度计算方法为Adam
    'MiniBatchSize', 64, ...            % 批训练 每次训练64个样本
    'MaxEpochs', 20, ...               % 最大训练次数为200次
    'InitialLearnRate', 0.001, ...      % 初始学习率为0.001
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...     % 调整后学习率为0.001 * 0.1
    'LearnRateDropPeriod', 150, ...     % 训练150次后 学习率进行调整
    'Shuffle', 'every-epoch', ...       % 每次训练打乱顺序
    'ValidationPatience', Inf, ...      % 关闭验证
    'Plots', 'training-progress', ...   % 画出训练曲线
    'ExecutionEnvironment', 'cpu', ...  % 采用CPU运行
    'Verbose', false);                  % 关闭命令行显示
%%  训练模型
net = trainNetwork(p_train, t_train, layers, options);

%%  模型预测
%t_sim1 = predict(net, p_train);
%t_sim2 = predict(net, p_test );


%%  数据反归一化
% T_sim1 = mapminmax('reverse', t_sim1', ps_output);
% T_sim2 = mapminmax('reverse', t_sim2', ps_output);

%%  绘制网络分析图
analyzeNetwork(layers)
