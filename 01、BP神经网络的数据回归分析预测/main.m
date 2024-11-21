%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('数据集.xlsx');

%%  划分训练集和测试集
temp = randperm(103);

P_train = res(temp(1: 80), 1: 7)';
T_train = res(temp(1: 80), 8)';
M = size(P_train, 2);

P_test = res(temp(81: end), 1: 7)';
T_test = res(temp(81: end), 8)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  创建网络
net = newff(p_train, t_train, [5]);

%%  设置训练参数
net.trainParam.epochs = 1000;     % 迭代次数 
net.trainParam.goal = 1e-6;       % 误差阈值
net.trainParam.lr = 0.01;         % 学习率

%%  训练网络
net= train(net, p_train, t_train);

%%  仿真测试
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  相关指标计算
%  R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
disp('********** R2 ************');
disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
disp('********** MAE ************');
disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;
disp('********** MBE ************');
disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2),])

%  RMSE
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
disp('********** RMSE ************');
disp(['训练集数据的RMSE为：', num2str(error1)])
disp(['测试集数据的RMSE为：', num2str(error2)])

%%  绘图预测图
figure

% 训练集预测结果子图
subplot(2, 1, 1) % 2行1列的布局，这是第一个子图
plot(1:M, T_train, 'r-*', 1:M, T_sim1, 'b-o', 'LineWidth', 1);
legend('真实值', '预测值');
xlabel('预测样本');
ylabel('预测结果');
string = {strcat('训练集预测结果对比：', ['RMSE=' num2str(error1)])};
title(string);
xlim([1, M]);
grid on;

% 测试集预测结果子图
subplot(2, 1, 2) % 2行1列的布局，这是第二个子图
plot(1:N, T_test, 'r-*', 1:N, T_sim2, 'b-o', 'LineWidth', 1);
legend('真实值', '预测值');
xlabel('预测样本');
ylabel('预测结果');
string = {strcat('测试集预测结果对比：', ['RMSE=' num2str(error2)])};
title(string);
xlim([1, N]);
grid on;


%% 绘制散点图
figure

sz = 25;
c = 'b';

% 训练集预测值 vs. 真实值 子图
subplot(2, 1, 1) % 2行1列的布局，这是第一个子图
scatter(T_train, T_sim1, sz, c);
hold on
plot([min(T_train), max(T_train)], [min(T_train), max(T_train)], '--k'); % 对角线
xlabel('训练集真实值');
ylabel('训练集预测值');
xlim([min(T_train) max(T_train)]);
ylim([min(T_sim1) max(T_sim1)]);
title('训练集预测值 vs. 训练集真实值');
grid on;
hold off;

% 测试集预测值 vs. 真实值 子图
subplot(2, 1, 2) % 2行1列的布局，这是第二个子图
scatter(T_test, T_sim2, sz, c);
hold on
plot([min(T_test), max(T_test)], [min(T_test), max(T_test)], '--k'); % 对角线
xlabel('测试集真实值');
ylabel('测试集预测值');
xlim([min(T_test) max(T_test)]);
ylim([min(T_sim2) max(T_sim2)]);
title('测试集预测值 vs. 测试集真实值');
grid on;
hold off;