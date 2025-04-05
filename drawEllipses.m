% Usage: plot_ellipses(Ellipses,size_im,fig_handle);
%
% Inputs: 
% Ellipses - parameters of the detected ellipses. Each coloumn contains
%                 [ x0 - x coordinate of the center of the ellipse
%                   y0 - y coordinate of the center of the ellipse
%                   a - length of semimajor axis
%                   b - length of semiminor axis
%                   alpha - angle of orientation of semimajor axis]
% size_im - size(im) where im is the gray image
% fig_handle - the handle of the figure if specified, if fig_handle=[] then
%                a new figure is created
%
% This function plots the ellipses
%
% Copyright (c) 2012 Dilip K. Prasad
% School of Computer Engineering
% Nanyang Technological University, Singapore
% http://www.ntu.edu.sg/
function [major_axes, minor_axes, centers, rotation_angles] = drawEllipses(ellipses_para, im)  
    if ~isempty(im)  
        % 显示图像  
        imshow(im, 'border', 'tight', 'initialmagnification', 'fit');  
        size_im = size(im);  
        hold on;  
    else  
        hold on;  
    end  
      
    % 初始化长轴数组  
    major_axes = zeros(1, size(ellipses_para, 2));  
    % 初始化短轴数组  
    minor_axes = zeros(1, size(ellipses_para, 2));  
    % 初始化圆心坐标数组  
    centers = zeros(2, size(ellipses_para, 2));  
    % 初始化旋转角度数组  
    rotation_angles = zeros(1, size(ellipses_para, 2));  
      
    th = 0:pi/180:2*pi;  
    for i = 1:size(ellipses_para, 2)  
        % 提取椭圆参数  
        Semi_major = ellipses_para(3, i); % 长轴  
        Semi_minor = ellipses_para(4, i); % 短轴  
        x0 = ellipses_para(1, i); % 中心x坐标  
        y0 = ellipses_para(2, i); % 中心y坐标  
        Phi = ellipses_para(5, i); % 旋转角度  
          
        % 计算椭圆上的点并绘制  
        x = x0 + Semi_major * cos(Phi) * cos(th) - Semi_minor * sin(Phi) * sin(th);  
        y = y0 + Semi_minor * cos(Phi) * sin(th) + Semi_major * sin(Phi) * cos(th);  
        plot(x, y, 'r', 'LineWidth', 2);  
          
        % 存储长轴长度、短轴长度、圆心坐标和旋转角度  
        major_axes(i) = Semi_major;  
        minor_axes(i) = Semi_minor;  
        centers(:, i) = [x0, y0];  
        rotation_angles(i) = Phi;  
    end  
      
    if ~isempty(im)  
        axis on;  
        set(gca, 'XTick', [], 'YTick', []);  
        axis ij;  
        axis equal;  
        axis([0 size_im(2) 0 size_im(1)]);  
    end  
      
    % 结束函数并输出长轴数组、短轴数组、圆心坐标数组和旋转角度数组  
end