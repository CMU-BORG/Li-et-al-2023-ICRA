close all
clear all

opts = detectImportOptions('FRR_Sept8th_DataSummary.tsv','FileType','delimitedtext');
SumT = readtable("FRR_Sept8th_DataSummary.tsv",opts);

ClosestDatalogs = string(SumT.ClosestDatalog);

timev = [0:0.01:120]';

% commandedGrasperAngle = zeros(tvl,3);
% move_to_pre_grasp = zeros(tvl,3);
% move_to_grasp =zeros(tvl,3);
% grasp =zeros(tvl,3);
% lift_after_grasp = zeros(tvl,3);
% move_to_grasp = zeros(tvl,3);
% move_to_pre_release = zeros(tvl,3);
% move_to_release = zeros(tvl,3);
% release = zeros(tvl,3);
% lift_after_release = zeros(tvl,3);

move_to_pre_grasp = [];
move_to_grasp = [];
grasp = [];


column_names_to_interpolate = {"x_m_","y_m_","z_m_","graspAngle_deg_","force","BufferLength","MoveToPreGrasp","MoveToGrasp","grasp","liftAfterGrasp","moveToPreRelease","moveToRelease","release","liftAfterRelease"};
interpType = {"linear","linear","linear","linear","previous","previous","previous","previous","previous","previous","previous","previous","previous","previous"};
TotTable = table();
InterpTable = table();

keySet = {'Jan','Feb','Mar','Apr'};
valueSet = [327.2 368.2 197.6 178.4];
M_transitions = containers.Map({'MoveToPreGrasp','MoveToGrasp','grasp','liftAfterGrasp','moveToPreRelease','moveToRelease','release','liftAfterRelease'},{[],[],[],[],[],[],[],[]});
M_transitions_sim = containers.Map({'MoveToPreGrasp','MoveToGrasp','grasp','liftAfterGrasp','moveToPreRelease','moveToRelease','release','liftAfterRelease'},{[],[],[],[],[],[],[],[]});


timingInfoMean=[]; 
timingInfoStd = [];
timingInfoDiff = [];

for k = 1:length(ClosestDatalogs)
    
    DataT = readtable(ClosestDatalogs(k));
    fileNameColumns = repmat(ClosestDatalogs(k),[size(DataT,1),1]);
    DataT.DatalogName = fileNameColumns;

    %interpolate the time-length
    %xdata = interpolateVal(timev,DataT.Time,DataT.x_m_,xdata,k); 
    

    InterpTable = vertcat(InterpTable, calculateInterpTable(timev,DataT,ClosestDatalogs(k),column_names_to_interpolate,interpType));

    %% get the point in time where the transitions occurred
    [M_transitions] = findTransitions(DataT,M_transitions);

    TotTable = vertcat(TotTable,DataT);
    timingInfoMean(k) = mean(diff(DataT.Time));
    timingInfoStd(k) = std(diff(DataT.Time));
    timingInfoDiff=[timingInfoDiff;diff(DataT.Time)];


end


grandMeanRealTiming = mean(timingInfoMean)
sampleStdRealTiming = std(timingInfoDiff)


interpMeanStd = calculateInterpMeanStd(InterpTable);

transitionAvgArr =[];
keyM = keys(M_transitions);
for jj = 1:length(keyM)

    mapval = M_transitions(keyM{jj});    
    transitionAvgArr(jj) = mean(mapval);


end

%%get Simulation results
SimDataT = readtable("SimResult.csv");
SimDataT=SimDataT(SimDataT.TimeReal>=0,:); %get rid of the trailing zeros.  The time real will be some large negative number after the official end of the experiment
SimDataT.Time = SimDataT.TimeReal;
Sim_time = SimDataT.TimeReal;
Sim_x = SimDataT.GantryHeadIndex_pos_x-SimDataT.GantryHeadIndex_pos_x(1);
Sim_y = -(SimDataT.BasePositionIndex_pos_y - SimDataT.BasePositionIndex_pos_y(1));
Sim_z = SimDataT.GantryHeadIndex_pos_z - SimDataT.GantryHeadIndex_pos_z(1);
Sim_graspQuat = table2array(SimDataT(:,{'ClawJointRightIndex_orient_k','ClawJointRightIndex_orient_x','ClawJointRightIndex_orient_y','ClawJointRightIndex_orient_z'}));
Sim_graspQuat(:,[1,2,3,4]) = Sim_graspQuat(:,[4,1,2,3]);
Sim_GrasperAngle=quat2axang(Sim_graspQuat);
Sim_GrasperAngle = rad2deg(Sim_GrasperAngle(:,4));
Sim_GrasperAngle = (Sim_GrasperAngle - Sim_GrasperAngle(100)).*2;

[M_transitions_sim] = findTransitions(SimDataT,M_transitions_sim);

keyM = keys(M_transitions_sim);
transitionAvgArr_sim =[];
for jj = 1:length(keyM)

    mapval = M_transitions_sim(keyM{jj});    
    transitionAvgArr_sim(jj) = mean(mapval);


end


diffSimTime= diff(Sim_time);
meanSimTime = mean(diffSimTime)
stdSimTime = std(diffSimTime)

%% Get Simulation Timing for SNS comparison
filetext = fileread('TimingProfile_Simulation_12thSept2022.txt');
sname=regexp(filetext,"SNS Time (.+)",'tokens','dotexceptnewline');
ssn = str2double([sname{:}]);
SNS_sim_time = mean(ssn)
SNS_std_time = std(ssn)


%get the loop time and compare to sim time
stIdx = regexp(filetext,"SNS Time (.+)",'dotexceptnewline');
textAfterSNS_starts = filetext(stIdx:length(filetext));
lname=regexp(textAfterSNS_starts,"LoopTime (.+)",'tokens','dotexceptnewline');
lsn = str2double([lname{:}]);
Loop_sim_time = mean(lsn)
loop_std_time = std(lsn)

%% Get Real-gantry Timing for SNS comparison
filetext = fileread('TimingProfile_7thSept_2022.txt');
sname=regexp(filetext,"SNS Time (.+)",'tokens','dotexceptnewline');
ssn = str2double([sname{:}]);
SNS_real_time = mean(ssn)
SNS_std_time = std(ssn)


%get the loop time and compare to sim time
stIdx = regexp(filetext,"SNS Time (.+)",'dotexceptnewline');
textAfterSNS_starts = filetext(stIdx:length(filetext));
lname=regexp(textAfterSNS_starts,"LoopTime: (.+)",'tokens','dotexceptnewline');
lsn = str2double([lname{:}]);
Loop_real_time = mean(lsn)
loop_std_time = std(lsn)



%% Plotting


figure()
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [ 2 2 6.4*2.5 4.7*2.5]);
set(gcf,'color','w');

set(0, 'DefaultAxesFontName', 'Arial')

tiledlayout(4,2)

dataNames={'x_m__','y_m__','z_m__','graspAngle_deg__'};
yaxislabels={"x-position (m)","y-position (m)","z-position (m)",{"Grasper","angle (deg)"}};
targetval=[0.15,0.15,-0.31];

%adjust grasp angle
interpMeanStd.graspAngle_deg__mean = interpMeanStd.graspAngle_deg__mean -interpMeanStd.graspAngle_deg__mean(1);

simDataPlot=[Sim_x,Sim_y,Sim_z,Sim_GrasperAngle];
axesC_real ={};  %to store axes objects
axesC_sim = {}; %to store axes objects for sim
lineC_real ={}; %to store line objects for actual data
lineC_sim = {}; %to store line objects for simulation

ax_xlim = [0 60]; %x axes limits
tval =(interpMeanStd.Time_mean>= ax_xlim(1) & interpMeanStd.Time_mean<= ax_xlim(2)); %done to prevent the lines from extending past the xlim.

for mm = 1:length(dataNames)
    axa = nexttile;
    hold on;
    dnm = dataNames{mm};
    lineC_real{end+1} = plot(axa,interpMeanStd.Time_mean(tval),interpMeanStd.(dnm+"mean")(tval),'k-','LineWidth',3,'DisplayName',"Experimental"+newline+"data");

    curve1 = interpMeanStd.(dnm+"std")+interpMeanStd.(dnm+"mean");
    curve2 = -interpMeanStd.(dnm+"std")+interpMeanStd.(dnm+"mean");
    timev2 =[timev(tval)',fliplr(timev(tval)')];
    inBetween = [curve1(tval)', fliplr(curve2(tval)')];
    lineC_real{end+1} =fill(timev2,inBetween,'cyan','FaceAlpha',0.2,"LineWidth",1,"LineStyle","--","EdgeColor",[0 0 1], "DisplayName","$$\pm 1 Std. dev.$$"); %https://www.mathworks.com/matlabcentral/answers/494515-plot-standard-deviation-as-a-shaded-area
    
    ylabel(yaxislabels{mm},"Rotation",0, 'VerticalAlignment','middle', 'HorizontalAlignment','right')
    

    

    set(axa,'FontSize',22,'FontWeight','bold','FontName','Arial');
    axesC_real{end+1} = axa;

    %Simulation results:
    axb = nexttile;
    hold on
    lineC_sim{end+1} = plot(Sim_time, simDataPlot(:,mm),'r-.','LineWidth',2,'DisplayName','Simulation');
    
    set(axb,'FontSize',22,'FontWeight','bold','FontName','Arial');
    axesC_sim{end+1} = axb;

    if mm~=length(dataNames)
        set(axa,'Xticklabel',[],'XColor','none') ;
        set(axb,'Xticklabel',[],'XColor','none') ;

    end


    if mm<4
        yline(axa,targetval(mm),'g-.','LineWidth',3,'DisplayName',"Target Position");
        yline(axb,targetval(mm),'g-.','LineWidth',3,'DisplayName',"Target Position");
    end
        
    
end

placeHolderSim = plot(axesC_real{1},0,0,'r-.','LineWidth',2,'DisplayName','Simulation'); %place holder sim line to get it on the same legend
legend(axesC_real{1},"Interpreter","latex")


axesC_real{4}.Clipping = "Off";
axesC_sim{4}.Clipping = "Off";

ylimv=([-10,40]); %for axes 4
ylim(axesC_real{4},ylimv);
ylim(axesC_sim{4},ylimv);

for jk = 1:length(transitionAvgArr)
    %xline(transitionAvgArr(jk),"k--", 'HandleVisibility','off');
    h = line(axesC_real{4},[transitionAvgArr(jk) transitionAvgArr(jk)],[-10 240],'LineStyle','--','Color','#99bab9','LineWidth',1);
    h2 = line(axesC_sim{4},[transitionAvgArr_sim(jk) transitionAvgArr_sim(jk)],[-10 240],'LineStyle','--','Color','#99bab9','LineWidth',1);
end





xlabel(axesC_real{4},"time (s)");
xlabel(axesC_sim{4},"time (s)");

%link axes
linkaxes([axesC_real{:}],'x');
xlim(axesC_real{4},[0 60]);

linkaxes([axesC_sim{:}],'x');
xlim(axesC_sim{4},[0 60]);

set(gcf, 'Renderer', 'painters');




function [M_transitions] = findTransitions(dataT,M_transitions)
    keyM = keys(M_transitions);
    for jj = 1:length(keyM)

        diffT = diff(dataT.(keyM{jj}));  
        idxT = find(diffT~=0); % Look for where there was a transition.  these indices are the value right before the change
        ActualVals = dataT.(keyM{jj}); 
        transitionT = dataT.Time(idxT(find((diffT(idxT)>=0 & ActualVals(idxT) == 0),1,'first'))); %get the index of the first transition from 0 to 20.



        mapval = M_transitions(keyM{jj});
        mapval(end+1) = transitionT;

        
        M_transitions(keyM{jj}) = mapval;



    end


end


function [interpMeanStd] = calculateInterpMeanStd(interpTable)
    columnNames = interpTable.Properties.VariableNames;
    
    interpMeanStd = table();

    G=findgroups(interpTable.Time);
    
    for jj=1:length(columnNames)
        columnName = columnNames{jj};
        if ~strcmp(class(interpTable.(columnName)),"string")
            averageV = splitapply(@mean,interpTable.(columnName),G);
            stdV = splitapply(@std,interpTable.(columnName),G);
            interpMeanStd.(columnName+"_mean") = averageV;
            interpMeanStd.(columnName+"_std") = stdV;
        end


    end





end


function [tempT] = calculateInterpTable(timev,DataT,datalogName,columnNamesToInterp,interpType)
    
    tempT = table();
    tempT.Time = timev;

    for jj = 1:length(columnNamesToInterp)
        columnName = columnNamesToInterp{jj};
        tempT.(columnName) = interpolateVal(timev,DataT.Time,DataT.(columnName),interpType{jj});

    end
    
    fileNameColumns = repmat(datalogName,[size(tempT,1),1]);
    tempT.DatalogName = fileNameColumns;

end


function [interp_vec] = interpolateVal(timev,origt,vec,interpType)
    origt =[0; origt];
    vec =[vec(1); vec];
    if max(origt)<max(timev)
        
        origt(end+1) = max(timev);
        vec(end+1) = vec(end);
    end
    interp_vec = interp1(origt,vec,timev,interpType);

end

function [interpMat] = RunningMeanStd(interp_vec,interpMat,n)

    prevMean = interpMat(:,2);
    interpMat(:,2) = prevMean + (interp_vec'-prevMean)./n; %mean
    interpMat(:,1) = interpMat(:,1) + (interp_vec'-interpMat(:,2)).*(interp_vec'-prevMean); %M2n according to Welford's Formula: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    interpMat(:,3) = (interpMat(:,1)/(max(1,n-1))).^(0.5);

    

end


