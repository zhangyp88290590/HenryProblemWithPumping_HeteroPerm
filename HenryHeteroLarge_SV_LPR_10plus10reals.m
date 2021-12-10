% This program is for generating the response surfaces of 3 state variables
% with different pumping rates (6) for 10 different realizations of the
% heterogeneous hydraulic conductivity (Hk) fields with variance (sigma^2)
% being 0.5 under 5 horizontal correlation length scales (lx)
%
% The 3 state variables are : 
% 1) the toe position (x) of the saltwater wedge along the bottom;
% 2) the concentration at the pumping well;
% 3) the total dissolved solute in the system.
%
% Author: Yipeng Zhang @ University of Texas at El Paso
% Date: 07/21/2021
clc
clear
close all
%--------------------------------------------------------------------------
%pump_rate = [0,1,2,3,4,5];

%pump_rate = [0,0.5,1,1.5,2]; % ajust the pumping rate
pump_rate = [0.0,0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0,5.0];% updated pumping rates to have 10 values
% with more pumping rates refined from 0.0 to 1.50

% Generate the response surface for the toe position 

A = zeros(10,10,5);
A_10 = zeros(10,10,5); % extra 10 reals for toe position (pumping rates, reals, range)
A_20 = zeros(10,20,5); % this is the matrix that stores all 20 reals for toe position


A2 = zeros(10,10,5);
A3 = zeros(10,10,5);


conc_wellMatrix = zeros(10,10,5);
salt_totMatrix = zeros(10,10,5);

conc_wellMatrix_10 = zeros(10,10,5); % extra 10 reals
salt_totMatrix_10 = zeros(10,10,5); % extra 10 reals

conc_wellMatrix_20 = zeros(10,20,5); % all 20 reals
salt_totMatrix_20 = zeros(10,20,5); % all 20 reals

conc_well = zeros(10,1);
salt_tot = zeros(10,1);

for i = 1:5 % the horizontal range loop
    folder = sprintf('HR%d',i-1);
    for j = 1:10 % the realization loop
        % comment 10/25/2021
        % there is a bug in the simulation code causing the hk0 folder is
        % not properly structured, need to fix later (i.e., j=1:10)
        folder2 = sprintf('hk%d',j-1);
        folder3 = 'sample3';
        filename_x = sprintf('toe_x.csv');
        filename_x1 = sprintf('toe_x1.csv');
        filename_x17p5 = sprintf('toe_x17p5.csv');
        
        filepath_x= fullfile(pwd,folder,folder2,folder3,filename_x);
        filepath_x1= fullfile(pwd,folder,folder2,folder3,filename_x1);
        filepath_x17p5= fullfile(pwd,folder,folder2,folder3,filename_x17p5);
        
        fid = fopen(filepath_x,'rt');
        formatSpec = '%f %f %f %f %f %f %f %f %f %f';
        sizeA = [10 Inf];
        A(:,j,i) = fscanf(fid,formatSpec,sizeA);
        fclose(fid);
        
        fid = fopen(filepath_x1,'rt');
        formatSpec = '%f %f %f %f %f ';
        sizeA = [10 Inf];
        A2(:,j,i) = fscanf(fid,formatSpec,sizeA);
        fclose(fid);
        
        fid = fopen(filepath_x17p5,'rt');
        formatSpec = '%f %f %f %f %f';
        sizeA = [10 Inf];
        A3(:,j,i) = fscanf(fid,formatSpec,sizeA);
        fclose(fid);
        
        %A = A';
        for k = 1:10 % pumping loop
            folder4 = sprintf('pump%d',k-1);
            %--------------------------------------------------------------
            % open the MT3D001.OBS file
            filename2 = sprintf('MT3D001.OBS');
            filepath2= fullfile(pwd,folder,folder2,folder3,folder4,filename2);
            fid2 = fopen(filepath2,'rt');
            tline = fgets(fid2);
            tline2 = fgets(fid2);
            formatSpec2 = '%f %f %f';
            sizeB = [3 Inf];
            B = fscanf(fid2,formatSpec2,sizeB);
            if B(3,end)<1e-10
                conc_well(k) = 0.0;
            else
                conc_well(k) = B(3,end);
            end
            fclose(fid2);
            %--------------------------------------------------------------
            % open the MT3D001.MAS file
            filename3 = sprintf('MT3D001.MAS');
            filepath3= fullfile(pwd,folder,folder2,folder3,folder4,filename3);
            fid3 = fopen(filepath3,'rt');
            tline = fgets(fid3);
            tline2 = fgets(fid3);
            formatSpec = '%f %f %f %f %f %f %f %f %f';
            sizeC = [9 Inf];
            C = fscanf(fid3,formatSpec,sizeC);
            %C = C';
            salt_tot(k) = C(7,end);
            fclose(fid3);
        end
        conc_wellMatrix(:,j,i) = conc_well;% assign the conentration array of each pumping rate 
                                           % to the storage matrix for jth
                                           % realization and i-th range
        salt_totMatrix(:,j,i) = salt_tot;% assign the salt_tot array of each pumping rate 
                                           % to the storage matrix for jth
                                           % realization and i-th range
    end
end

%--------------------------------------------------------------------------
% load additional data from 10 extra realizations
% "H:\HenryProblem_GPRSurrogate\HeterHk_86p4_1_lowPR_10extra"
folder_10reals = "H:\HenryProblem_GPRSurrogate\HeterHk_86p4_0p5_lowPR_10extra";

for i = 1:5 % the horizontal range loop
    folder = sprintf('HR%d',i-1);
    for j = 1:10 % the realization loop
        % comment 10/25/2021
        % there is a bug in the simulation code causing the hk0 folder is
        % not properly structured, need to fix later (i.e., j=1:10)
        folder2 = sprintf('hk%d',j-1);
        folder3 = 'sample3';
        filename_x = sprintf('toe_x.csv');
        filename_x1 = sprintf('toe_x1.csv');
        filename_x17p5 = sprintf('toe_x17p5.csv');
        
        filepath_x= fullfile(folder_10reals,folder,folder2,folder3,filename_x);
        filepath_x1= fullfile(folder_10reals,folder,folder2,folder3,filename_x1);
        filepath_x17p5= fullfile(folder_10reals,folder,folder2,folder3,filename_x17p5);
        
        fid = fopen(filepath_x,'rt');
        formatSpec = '%f %f %f %f %f %f %f %f %f %f';
        sizeA = [10 Inf];
        A_10(:,j,i) = fscanf(fid,formatSpec,sizeA);
        fclose(fid);
        
        fid = fopen(filepath_x1,'rt');
        formatSpec = '%f %f %f %f %f';
        sizeA = [10 Inf];
        A2(:,j,i) = fscanf(fid,formatSpec,sizeA);
        fclose(fid);
        
        fid = fopen(filepath_x17p5,'rt');
        formatSpec = '%f %f %f %f %f';
        sizeA = [10 Inf];
        A3(:,j,i) = fscanf(fid,formatSpec,sizeA);
        fclose(fid);
        
        %A = A';
        for k = 1:10 % pumping loop
            folder4 = sprintf('pump%d',k-1);
            %--------------------------------------------------------------
            % open the MT3D001.OBS file
            filename2 = sprintf('MT3D001.OBS');
            filepath2= fullfile(folder_10reals,folder,folder2,folder3,folder4,filename2);
            fid2 = fopen(filepath2,'rt');
            tline = fgets(fid2);
            tline2 = fgets(fid2);
            formatSpec2 = '%f %f %f';
            sizeB = [3 Inf];
            B = fscanf(fid2,formatSpec2,sizeB);
            if B(3,end)<1e-10
                conc_well(k) = 0.0;
            else
                conc_well(k) = B(3,end);
            end
            fclose(fid2);
            %--------------------------------------------------------------
            % open the MT3D001.MAS file
            filename3 = sprintf('MT3D001.MAS');
            filepath3= fullfile(folder_10reals,folder,folder2,folder3,folder4,filename3);
            fid3 = fopen(filepath3,'rt');
            tline = fgets(fid3);
            tline2 = fgets(fid3);
            formatSpec = '%f %f %f %f %f %f %f %f %f';
            sizeC = [9 Inf];
            C = fscanf(fid3,formatSpec,sizeC);
            %C = C';
            salt_tot(k) = C(7,end);
            fclose(fid3);
        end
        conc_wellMatrix_10(:,j,i) = conc_well;% assign the conentration array of each pumping rate 
                                           % to the storage matrix for jth
                                           % realization and i-th range
        salt_totMatrix_10(:,j,i) = salt_tot;% assign the salt_tot array of each pumping rate 
                                           % to the storage matrix for jth
                                           % realization and i-th range
    end
end


% concatenate the first 10 reals with the extra 10 reals
for i = 1:size(A_20,3)
    
    A_20(:,:,i) = [A(:,:,i) A_10(:,:,i)];
    conc_wellMatrix_20(:,:,i) = [conc_wellMatrix(:,:,i) conc_wellMatrix_10(:,:,i)];
    salt_totMatrix_20(:,:,i) = [salt_totMatrix(:,:,i) salt_totMatrix_10(:,:,i)];
    
end

% find the mediam values of the three State Variables
A_20_md = median(A_20,2);
conc_well20_md = median(conc_wellMatrix_20,2);
salt_tot20_md = median(salt_totMatrix_20,2);

% find the mean values of the three State Variables
A_20_mean = mean(A_20,2);
conc_well20_mean = mean(conc_wellMatrix_20,2);
salt_tot20_mean = mean(salt_totMatrix_20,2);


%--------------------------------------------------------------------------
% load the S.V. of the homogeneous case (hk =86.4) from txt files

% load the toe postion
filename = 'toeX_86p4Homo.txt';
fid = fopen(filename,'rt');
formatSpec = '%f %f %f %f %f %f';
size_toeXHomo = [6 Inf];
toeXHomo = zeros(6,1);
toeXHomo(:,:) = fscanf(fid,formatSpec,size_toeXHomo);
fclose(fid);

% load the well concentration
filename = 'wellConc_86p4Homo.txt';
fid = fopen(filename,'rt');
formatSpec = '%f %f %f %f %f %f';
size_wellConc = [6 Inf];
wellConcHomo = zeros(6,1);
wellConcHomo(:,:) = fscanf(fid,formatSpec,size_wellConc);
fclose(fid);

% load the total mass
filename = 'totMass_86p4Homo.txt';
fid = fopen(filename,'rt');
formatSpec = '%f %f %f %f %f %f';
size_totMass = [6 Inf];
totMassHomo = zeros(6,1);
totMassHomo(:,:) = fscanf(fid,formatSpec,size_totMass);
fclose(fid);



% Horizontal range array 
HR_arr = [10,40,70,100,200];

% Generate the response surface for the toe position

for i = 1:5
    figure(i)
    plot(pump_rate,A_20(:,:,i),'o-');
    hold on
%     plot(pump_rate,toeXHomo,'k*-','LineWidth',1);
%     hold on
    plot(pump_rate, A_20_md(:,:,i),'r*-','LineWidth',1);
    hold on
    plot(pump_rate, A_20_mean(:,:,i),'b*-','LineWidth',1);
    ylim([30 200])
    xlabel('normalized pumping rate w.r.t. Qin');
    ylabel('distance from the inland boundary (m)');
    title(['Toe Position along the Bottom V.S. Pumping Rate H-Range = ',num2str(HR_arr(i)),'m']);
    legend('real0','real1','real2','real3','real4','real5','real6','real7','real8','real9',...
        'real10','real11','real12','real13','real14','real15','real16','real17','real18',...
        'real19');
    saveas(figure(i),fullfile(pwd, 'OutPutFigs','20reals',sprintf('20reals_toeX_HR%d.png',i-1)));
    close(figure(i))
end

%---------------------log scale--------------------------------------------
% Generate the response surface for the toe position
%---------------------log scale--------------------------------------------
for i = 1:5
    figure(i)
    semilogy(pump_rate,A_20(:,:,i),'o-');
    hold on
%     plot(pump_rate,toeXHomo,'k*-','LineWidth',1);
%     hold on
    semilogy(pump_rate, A_20_md(:,:,i),'r*-','LineWidth',1);
    hold on
    semilogy(pump_rate, A_20_mean(:,:,i),'b*-','LineWidth',1);
    ylim([30 200])
    xlabel('normalized pumping rate w.r.t. Qin');
    ylabel('distance from the inland boundary (m)');
    title(['Toe Position along the Bottom V.S. Pumping Rate H-Range = ',num2str(HR_arr(i)),'m']);
    legend('real0','real1','real2','real3','real4','real5','real6','real7','real8','real9',...
        'real10','real11','real12','real13','real14','real15','real16','real17','real18',...
        'real19');
    saveas(figure(i),fullfile(pwd, 'OutPutFigs','20reals',sprintf('20reals_toeX_Log_HR%d.png',i-1)));
    close(figure(i))
end



% Generate the response surface for the well concentration

for i = 1:5
    figure(i)
    plot(pump_rate,conc_wellMatrix_20(:,:,i),'o-');
    hold on
%     plot(pump_rate,wellConcHomo,'k*-','LineWidth',1);
%     hold on
    plot(pump_rate, conc_well20_md(:,:,i),'r*-','LineWidth',1);
    hold on
    plot(pump_rate, conc_well20_mean(:,:,i),'b*-','LineWidth',1);
    %ylim([50 200])
    xlabel('normalized pumping rate w.r.t. Qin');
    ylabel('Conc. at the Well (g/L)');
    title(['Concentration at the Well V.S. Pumping Rate H-Range = ',num2str(HR_arr(i)),'m']);
    legend('real0','real1','real2','real3','real4','real5','real6','real7','real8','real9',...
        'real10','real11','real12','real13','real14','real15','real16','real17','real18',...
        'real19','Location','southeast');
    saveas(figure(i),fullfile(pwd, 'OutPutFigs','20reals',sprintf('20reals_concWell_HR%d.png',i-1)));
    close(figure(i))
end


%---------------------log scale--------------------------------------------
% Generate the response surface for the well concentration
%---------------------log scale--------------------------------------------
for i = 1:5
    figure(i)
    semilogy(pump_rate,conc_wellMatrix_20(:,:,i),'o-');
    hold on
%     plot(pump_rate,wellConcHomo,'k*-','LineWidth',1);
%     hold on
    semilogy(pump_rate, conc_well20_md(:,:,i),'r*-','LineWidth',1);
    hold on
    semilogy(pump_rate, conc_well20_mean(:,:,i),'b*-','LineWidth',1);
    %ylim([50 200])
    xlabel('normalized pumping rate w.r.t. Qin');
    ylabel('Conc. at the Well (g/L)');
    title(['Concentration at the Well V.S. Pumping Rate H-Range = ',num2str(HR_arr(i)),'m']);
    legend('real0','real1','real2','real3','real4','real5','real6','real7','real8','real9',...
        'real10','real11','real12','real13','real14','real15','real16','real17','real18',...
        'real19','Location','southeast');
    saveas(figure(i),fullfile(pwd, 'OutPutFigs','20reals',sprintf('20reals_concWell_Log_HR%d.png',i-1)));
    close(figure(i))
end

% Generate the response surface for the total dissolved salt in the system

for i = 1:5
    figure(i)
    plot(pump_rate,salt_totMatrix_20(:,:,i),'o-');
    hold on
%     plot(pump_rate,totMassHomo,'k*-','LineWidth',1);
%     hold on
    plot(pump_rate,salt_tot20_md(:,:,i),'r*-','LineWidth',1);
    hold on
    plot(pump_rate,salt_tot20_mean(:,:,i),'b*-','LineWidth',1);
    ylim([0 18e4])
    xlabel('normalized pumping rate w.r.t. Qin');
    ylabel('Total Dissolved Mass (kg)');
    title(['Total Dissolved Mass V.S. Pumping Rate H-Range = ',num2str(HR_arr(i)),'m']);
    legend('real0','real1','real2','real3','real4','real5','real6','real7','real8','real9',...
        'real10','real11','real12','real13','real14','real15','real16','real17','real18',...
        'real19','Location','southeast');
    
    saveas(figure(i),fullfile(pwd, 'OutPutFigs','20reals',sprintf('20reals_totSalt_HR%d.png',i-1)));
    close(figure(i))
end

%---------------------log scale--------------------------------------------
% Generate the response surface for the total dissolved salt in the system
%---------------------log scale--------------------------------------------
for i = 1:5
    figure(i)
    semilogy(pump_rate,salt_totMatrix_20(:,:,i),'o-');
    hold on
%     plot(pump_rate,totMassHomo,'k*-','LineWidth',1);
%     hold on
    semilogy(pump_rate,salt_tot20_md(:,:,i),'r*-','LineWidth',1);
    hold on
    semilogy(pump_rate,salt_tot20_mean(:,:,i),'b*-','LineWidth',1);
    ylim([0 18e4])
    xlabel('normalized pumping rate w.r.t. Qin');
    ylabel('Total Dissolved Mass (kg)');
    title(['Total Dissolved Mass V.S. Pumping Rate H-Range = ',num2str(HR_arr(i)),'m']);
    legend('real0','real1','real2','real3','real4','real5','real6','real7','real8','real9',...
        'real10','real11','real12','real13','real14','real15','real16','real17','real18',...
        'real19','Location','southeast');
    
    saveas(figure(i),fullfile(pwd, 'OutPutFigs','20reals',sprintf('20reals_totSalt_Log_HR%d.png',i-1)));
    close(figure(i))
end