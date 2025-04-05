function []  = LCS_ellipse()
%% parameters illustration
%1) Tac: 
%The threshold of elliptic angular coverage which ranges from 0~360. 
%The higher Tac, the more complete the detected ellipse should be.
%2) Tr:
%The ratio of support inliers to ellipse which ranges from 0~1.
%The higher Tr, the more sufficient the support inliers are.
%3) specified_polarity: 
%1 means detecting the ellipses with positive polarity;
%-1 means detecting the ellipses with negative polarity; 
%0 means detecting all ellipses from image
clear;
close all;
clc
%image path

filename = '2.png';

% parameters
Tac = 100;
Tr = 0.6;
specified_polarity = 0;

%%
% read image 
disp('------read image------');
I = imread(filename);


%% detecting ellipses from real-world images
[ellipses, ~, posi] = ellipseDetectionByArcSupportLSs(I, Tac, Tr, specified_polarity);

disp('draw detected ellipses');
drawEllipses(ellipses',I);
% display
ellipses(:,5) = ellipses(:,5)./pi*180;
disp(['The total number of detected ellipses��',num2str(size(ellipses,1))]);
[major_axes_lengths, minor_axes_lengths, centers_output, rotation_angles_output] = drawEllipses(ellipses', I); 
% ��ʾ���᳤��  
disp('���᳤��:');  
disp(major_axes_lengths);  
  
% ��ʾԲ������  
disp('Բ������:');  
disp(centers_output);

% ��ʾ���᳤��  
disp('���᳤��:');  
disp(minor_axes_lengths);  

% ��ʾ��ת�Ƕ�  
disp('��ת�Ƕ�:');  
disp(rotation_angles_output);


%% draw ellipse centers
hold on;
candidates_xy = round(posi+0.5);%candidates' centers (col_i, row_i)
plot(candidates_xy(:,1),candidates_xy(:,2),'.');%draw candidates' centers.

%% write the result image
set(gcf,'position',[0 0 size(I,2) size(I,1)]);
% Define the path to the 'run' folder  
    runFolderPath = fullfile('pics'); 
      
    % Define the filename and path for saving the result  
    resultFilename = fullfile(runFolderPath, '101.jpg'); %�����Ǹķ�λ�õ� 
      
    % Save the figure as a JPG image to the 'run' folder  
    saveas(gcf, resultFilename, 'jpg');   %�����ǸĴ�λ�õ�
      
    % Close the figure window (optional)  
    close(gcf);  
    
    % End the function 
end



