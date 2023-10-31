from .GantrySimulation import GantrySimulation
import pybullet as p
import time
import math
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

gS=GantrySimulation()
#add object to the simulation at the center of the plate
gS.addObjectsToSim("PickupCube",startPos=[0,0,(0.063+0.02)],mass_kg=0.2,sizeScaling=0.6,
                   sourceFile=str(pathlib.Path.cwd()/"envs\\URDF\\PickUpObject_URDF\\urdf\\PickUpObject_URDF.urdf"))

while (not gS.CheckStopSim()):  # check to see if the button was pressed to close the sim
    #--------create sinusoidal movement of all the axes between their min and max positions-------
    ts = gS.timeStep #time step of the simulation in seconds
    nsteps = gS.simCounter #of simulation steps taken so far
    periodT = 4 #take 4 seconds to complete one sinusoid cycle

    #calculate x position
    jointInfo = gS.bulletClient.getJointInfo(gS.gantryId,gS.GantryLinkIndex_dict["GantryHeadIndex"]) #{"BasePositionIndex":6,"ZAxisBarIndex":1,"GantryHeadId":2,"ClawJointLeftIndex":3,"ClawJointRightIndex":4})
    lowerLimit = jointInfo[8] #getJointInfo does not scale the joint limits properly even when the URDF is imported with a scale factor. While this works out because scaling is applied in StepSim, bear this in mind.
    upperLimit = jointInfo[9] #same as above
    x = 0.5*(upperLimit-lowerLimit)*np.sin(2*np.pi*nsteps*ts/periodT)  + 0.5*(upperLimit-lowerLimit) + lowerLimit

    #calculate y position
    jointInfo = gS.bulletClient.getJointInfo(gS.gantryId, gS.GantryLinkIndex_dict["BasePositionIndex"])  # {"BasePositionIndex":6,"ZAxisBarIndex":1,"GantryHeadId":2,"ClawJointLeftIndex":3,"ClawJointRightIndex":4})
    lowerLimit = jointInfo[8]  # getJointInfo does not scale the joint limits properly even when the URDF is imported with a scale factor. While this works out because scaling is applied in StepSim, bear this in mind.
    upperLimit = jointInfo[9]  # same as above
    y = 0.5 * (upperLimit - lowerLimit) * np.sin(2 * np.pi * nsteps * ts / periodT) + 0.5 * (
                upperLimit - lowerLimit) + lowerLimit

    # calculate z position
    jointInfo = gS.bulletClient.getJointInfo(gS.gantryId, gS.GantryLinkIndex_dict["ZAxisBarIndex"])
    lowerLimit = jointInfo[8]  # getJointInfo does not scale the joint limits properly even when the URDF is imported with a scale factor. While this works out because scaling is applied in StepSim, bear this in mind.
    upperLimit = jointInfo[9]  # same as above
    z = 0.5 * (upperLimit - lowerLimit) * np.sin(2 * np.pi * nsteps * ts / periodT) + 0.5 * (
                upperLimit - lowerLimit) + lowerLimit

    # calculate claw opening
    # {"BasePositionIndex":6,"ZAxisBarIndex":1,"GantryHeadId":2,"ClawJointLeftIndex":3,"ClawJointRightIndex":4})
    jointInfo = gS.bulletClient.getJointInfo(gS.gantryId, gS.GantryLinkIndex_dict["ClawJointLeftIndex"])
    lowerLimit = jointInfo[8]  # should be in rad
    upperLimit = jointInfo[9]  # same as above
    leftGrasper_Rad = 0.5 * (upperLimit - lowerLimit) * np.sin(2 * np.pi * nsteps * ts / periodT) + 0.5 * (
            upperLimit - lowerLimit) + lowerLimit

    jointInfo = gS.bulletClient.getJointInfo(gS.gantryId, gS.GantryLinkIndex_dict["ClawJointRightIndex"])
    lowerLimit = jointInfo[8]  # should be in rad
    upperLimit = jointInfo[9]  # same as above
    rightGrasper_Rad = -0.5 * (upperLimit - lowerLimit) * np.sin(2 * np.pi * nsteps * ts / periodT) + 0.5 * (
            upperLimit - lowerLimit) + lowerLimit  #note that the right grasper should have a negative position command.  The left grasper is positive.

    ArgumentDict={"x_gantryHead":x,"y_BasePos":y,"z_AxisBar":z,"x_force":50,"y_force":500,"z_force":500,"leftGrasper_Rad":leftGrasper_Rad,
            "rightGrasper_Rad":rightGrasper_Rad,"leftGrasperTorque":25,"rightGrasperTorque":25}

    #---------step the simulation----------
    gS.stepSim(usePositionControl=True, GUI_override = True, **ArgumentDict) #pass argument dict to function