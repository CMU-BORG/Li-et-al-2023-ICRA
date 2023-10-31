import pybullet as p
import time
import pathlib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import controller.torchSNS as tSNS
from controller.torchSNS.torch import SNSCell
from controller.SNS_layer import perceptor, controller
from envs.GantrySimulation import GantrySimulation

#########################################################

def pick_and_place():
    gS = GantrySimulation()
    # add object to the simulation at the center of the plate
    gS.addObjectsToSim("PickupCube", startPos=[0, 0, (0.063+0.02)], mass_kg=0.2, sizeScaling=0.6,
                       sourceFile=str(pathlib.Path.cwd()/"envs\\URDF\\PickUpObject_URDF\\urdf\\PickUpObject_URDF.urdf"))

    GUI_control = True

    HeaderStr =  [ "Move to Pre Grasp",
         "Move to grasp"
            , "grasp", "lift after grasp", "move to pre release", "move to release", "release", "lift after release"]

    logArray = []


    while (not gS.CheckStopSim()):  # check to see if the button was pressed to close the sim
        timeStart=time.perf_counter()

        GUIcontrolTarget = gS.bulletClient.readUserDebugParameter(
            gS.GUIcontrols["GUIcontrolId"])
        if GUIcontrolTarget % 2 == 0 and GUI_control is True:
            GUI_control = False
            gS.simCounter = 0
            object_position = torch.Tensor([0, 0, -0.305]).unsqueeze(dim=0)
            target_position = torch.Tensor(
                [0.15, 0.15, -0.310]).unsqueeze(dim=0)

        ts = gS.timeStep  # time step of the simulation in seconds
        nsteps = gS.simCounter  # of simulation steps taken so far
        timev = ts*nsteps
        x = gS.bulletClient.getJointState(
            gS.gantryId, gS.GantryLinkIndex_dict["GantryHeadIndex"])[0]
        y = gS.bulletClient.getJointState(
            gS.gantryId, gS.GantryLinkIndex_dict["BasePositionIndex"])[0]
        z = gS.bulletClient.getJointState(
            gS.gantryId, gS.GantryLinkIndex_dict["ZAxisBarIndex"])[0]
        left_angle = gS.bulletClient.getJointState(
            gS.gantryId, gS.GantryLinkIndex_dict["ClawJointLeftIndex"])[0]
        right_angle = gS.bulletClient.getJointState(
            gS.gantryId, gS.GantryLinkIndex_dict["ClawJointRightIndex"])[0]
        force_feedback = gS.bulletClient.getContactPoints(
            gS.gantryId, gS.objects["PickupCube"].objId, gS.GantryLinkIndex_dict["ClawJointLeftIndex"], -1)
        if len(force_feedback) != 0:
            force = force_feedback[0][9]
        else:
            force = 0
        gripper_position = torch.Tensor([x, y, z]).unsqueeze(dim=0)
        force = torch.Tensor([force]).unsqueeze(dim=0)

        if GUI_control is False:
            tic = time.perf_counter()
            commands = perceptor.forward(
                gripper_position, object_position, target_position, force)
            [move_to_pre_grasp, move_to_grasp, grasp, lift_after_grasp, move_to_pre_release,
                move_to_release, release, lift_after_release] = commands.squeeze(dim=0).numpy()
            [x_d, y_d, z_d, leftGrasper_Rad_d, rightGrasper_Rad_d] = controller.forward(
                object_position, target_position, commands).numpy()
            if lift_after_release > 10:
                object_position = torch.Tensor([0, 0, 0]).unsqueeze(dim=0)

            toc = time.perf_counter()
            print("SNS Time " + str(toc - tic))

            logStr =   [ move_to_pre_grasp, move_to_grasp, grasp,
                                lift_after_grasp, move_to_pre_release,
                                move_to_release, release, lift_after_release]
            logArray.append(logStr)

        else:
            [x_d, y_d, z_d, leftGrasper_Rad_d, rightGrasper_Rad_d] = [0, 0, 0, 0, 0]


        print("xyz:"+str([x_d,y_d,z_d]))
        ArgumentDict = {"x_gantryHead": x_d, "y_BasePos": y_d, "z_AxisBar": z_d, "x_force": 50, "y_force": 500, "z_force": 500,
                        "leftGrasper_Rad": leftGrasper_Rad_d,"rightGrasper_Rad": rightGrasper_Rad_d, "leftGrasperTorque": 25,
                        "rightGrasperTorque": 25,"maxVelocity_x":0.04*1.8, "maxVelocity_y":0.0183*1.8,"maxVelocity_z":0.0349*3.15}




        # ---------step the simulation----------
        gS.stepSim(usePositionControl=True, GUI_override=False, **
                   ArgumentDict)  # pass argument dict to function
        timeStop = time.perf_counter()
        print("LoopTime " + str(timeStart - timeStop))

    #log to file:
    logDict = {}
    for (k,v) in gS.LinkPosition_dict.items():
        logDict[k+"_pos_x"]=v[:,0]
        logDict[k + "_pos_y"] = v[:, 1]
        logDict[k + "_pos_z"] = v[:, 2]

    for (k,v) in gS.LinkOrientation_dict.items():
        logDict[k+"_orient_x"]=v[:,0]
        logDict[k + "_orient_y"] = v[:, 1]
        logDict[k + "_orient_z"] = v[:, 2]
        logDict[k + "_orient_k"] = v[:, 3]


    logDict["Time"] = gS.timelog
    logDict["TimeReal"] = [x-gS.timelog_real[0] for x in gS.timelog_real]
    DF =pd.DataFrame.from_dict(logDict)

    neuronDF = pd.DataFrame.from_dict(logArray)
    neuronDF.columns = HeaderStr

    DF = DF.join(neuronDF)


    DF.to_csv("Data\\SimResult.csv")


    print("Finished")


if __name__ == "__main__":
    pick_and_place()

