import pybullet as p
import time
import pathlib
import numpy as np
import torch
import torch.nn as nn
import controller.torchSNS as tSNS
import re
import serial
from controller.torchSNS.torch import SNSCell
from controller.SNS_layer import perceptor, controller
from envs.GantrySimulation import GantrySimulation

#########################################################

def sendCommand(ser,commandstr,wait_for_ok = True):
    ser.write(commandstr.encode('utf-8'))
    # readval = ser.read(20)

    # print(readval)
    line_hold=""
    while wait_for_ok:
        line = ser.readline()
        #print(line)
        line_hold = line_hold + str(line, 'utf-8') +"\n"

        if line == b'ok\n':
            break

    return(line_hold)



def SetupGantry(comport='COM3',serialRate=115200,timeout=2,initPos=[220,220,315]):
    ser = serial.Serial(comport, serialRate, timeout=timeout)
    time.sleep(20)
    sendCommand(ser, "G28 X0 Y0 Z0\r\n")
    # sendCommand(ser,"G0 F15000 X0\r\n")
    sendCommand(ser, "M400\r\n")
    time.sleep(2)
    sendCommand(ser, "G90\r\n")
    print("Finished Sending G90")
    time.sleep(1)
    sendCommand(ser, "G0 F15000\r\n")
    time.sleep(1)
    sendCommand(ser, "G0 F15000 "+ 'X{0} Y{1} Z{2}'.format(*initPos)+"\r\n")
    sendCommand(ser, "M400\r\n") #waits until all motions in the planning queue are completed
    time.sleep(1)
    sendCommand(ser,"M280 P0 S95\r\n") #make sure that the gripper is closed
    sendCommand(ser,"M400\r\n")

    return(ser)



def pick_and_place(log_name =""):

    gS = GantrySimulation()
    # add object to the simulation at the center of the plate
    gS.addObjectsToSim("PickupCube", startPos=[0, 0, (0.063+0.02)], mass_kg=0.2, sizeScaling=0.6,
                       sourceFile=str(pathlib.Path.cwd()/"envs\\URDF\\PickUpObject_URDF\\urdf\\PickUpObject_URDF.urdf"))

    GUI_control = True

    x,y,z,left_angle,right_angle = 0,0,0,0,0 #initialize_to_zero
    x_off,y_off,z_off,left_angle_off,right_angle_off = 220, 220, 315, 0,0 #offsets for each of the axes
    force = 0


    maxFeedRate = 150 * 60  # mm/min
    periodT = 25  # seconds.  Done to violate maximum speed of 150 mm/sec
    accel = 300.48  # mm/s for x axis
    d_total = 400  # mm

    ts = 0  # time step of the simulation in seconds

    logArray = []

    BeginSNS_T = 0
    delT = 0



    while (not gS.CheckStopSim()):  # check to see if the button was pressed to close the sim

        startt_loop = time.time()
        GUIcontrolTarget = gS.bulletClient.readUserDebugParameter(
            gS.GUIcontrols["GUIcontrolId"])
        if GUIcontrolTarget % 2 == 0 and GUI_control is True:
            GUI_control = False
            gS.simCounter = 0
            object_position_array = [0, 0, -0.305]
            object_position = torch.Tensor(object_position_array).unsqueeze(dim=0)
            target_position_array = [0.15, 0.15, -0.310]
            target_position = torch.Tensor(target_position_array).unsqueeze(dim=0)
            BeginSNS_T = time.time()

            #for log file append
            logArray.append("Current Date: "+time.strftime("%a %d %b %Y %H:%M:%S",time.localtime()))
            logArray.append("Object Position (x,y,z)(m): "+",".join([str(x) for x in object_position_array]))
            logArray.append("Target Position (x,y,z)(m): " + ",".join([str(x) for x in target_position_array]))


            HeaderStr = ",".join(["Time","x (m)","y (m)","z (m)","grasp angle (deg)", "force","Buffer Length","Move to Pre Grasp", "Move to grasp"
                         ,"grasp","lift after grasp","move to pre release", "move to release", "release", "lift after release"])
            logArray.append(HeaderStr)




        #got current x,y and z positions via M114 command
        try:
            tic = time.perf_counter()
            responseM114 = sendCommand(ser, "M114 B\r\n")
            M114_restr= "X:(?P<Xcom>\d*\.\d*).*Y:(?P<Ycom>\d*\.\d*).*Z:(?P<Zcom>\d*\.\d*).*E:(?P<Ecom>\d*\.\d*).*B:(?P<BufferCountHead>\d*)\s*(?P<BufferCountTail>\d*).*Count.*X:(?P<Xsteps>\d*).*Y:(?P<Ysteps>\d*).*Z:(?P<Zsteps>\d*)"
            matchM114 = re.search(M114_restr, responseM114)
            matchM114_dict = matchM114.groupdict()
            Xcom,Ycom,Zcom,Ecom = float(matchM114_dict['Xcom']),float(matchM114_dict['Ycom']),float(matchM114_dict['Zcom']),float(matchM114_dict['Ecom']) #not real time positons, but what was commanded last
            Xsteps, Ysteps, Zsteps = int(matchM114_dict['Xsteps']),int(matchM114_dict['Ysteps']),int(matchM114_dict['Zsteps']) #position in steps
            BufferCountHead = int(matchM114_dict['BufferCountHead']) #buffer size is 16 (0 indexed, 0:15)
            BufferCountTail = int(matchM114_dict['BufferCountTail'])
            BufferLength = (BufferCountHead-BufferCountTail if BufferCountHead>=BufferCountTail else BufferCountHead-BufferCountTail+16)
            x,y,z = (Xsteps/80 - x_off)/1000 , (Ysteps/80 - y_off)/1000 , (Zsteps/400 - z_off)/1000 #convert from steps to mm
            toc = time.perf_counter()
            print("M114 Time "+str(toc-tic))
        except Exception as e:
            print(responseM114)

        #print ([x,y,z])

        gripper_position = torch.Tensor([x, y, z]).unsqueeze(dim=0)
        force = torch.Tensor([force]).unsqueeze(dim=0)

        #get SNS commands and send to printer
        if GUI_control is False:
            tic =time.perf_counter()
            commands = perceptor.forward(
                gripper_position, object_position, target_position, force)
            [move_to_pre_grasp, move_to_grasp, grasp, lift_after_grasp, move_to_pre_release,
                move_to_release, release, lift_after_release] = commands.squeeze(dim=0).numpy()
            [x_d, y_d, z_d, leftGrasper_Rad_d, rightGrasper_Rad_d] = controller.forward(
                object_position, target_position, commands).numpy()

            #print([x_d,y_d,z_d])
            if lift_after_release > 10:
                object_position = torch.Tensor([0, 0, 0]).unsqueeze(dim=0)
            toc = time.perf_counter()
            print("SNS Time " + str(toc - tic))
            # send commands

            tic = time.perf_counter()
            graspAngle = abs(np.interp(abs(np.rad2deg(rightGrasper_Rad_d)),[0,45],[95,50])) #servo fully closed is 95 degrees, servo open is 50 degrees.  Sim closed is 0 degrees, sim open is 45 degrees
            posvec = [50*60, x_d*1000+x_off, y_d*1000+y_off,z_d*1000+z_off] #first element is the speed in mm/min

            commandStr = "G0 F{0:.2f} X{1:.2f} Y{2:.2f} Z{3:.2f} \r\n".format(*posvec)
            grasperStr = "M280 P0 S{0:.2f} \r\n".format(graspAngle)
            if BufferLength<10: #only send commands when the buffer is small
                #print("Buffer "+str(BufferLength))
                #print(commandStr)
                sendCommand(ser,commandStr)
            toc = time.perf_counter()
            print("G0 movement " + str(toc - tic))

            #sendCommand(ser, "M400\r\n")  # waits until all motions in the planning queue are completed
            #assume actual position is the same as the commanded position
            #print(grasperStr)
            tic = time.perf_counter()
            if right_angle != graspAngle: #M280 takes time to execute, so just send M280 if there is a new angle to move to
                sendCommand(ser,grasperStr)
            right_angle = graspAngle
            #ser.reset_input_buffer() #get rid of what is in the buffer
            #sendCommand(ser,"M400\r\n")
            toc = time.perf_counter()
            print("M280 movement " + str(toc - tic))



            #check for contact
            tic = time.perf_counter()
            responseContact = sendCommand(ser,"M43 I P2\r\n")
            re_str = 'PIN:.*Port:.*Input\\s*=\\s*(?P<ReadValue>\\d).*'
            matchV = re.search(re_str,responseContact)
            #print(matchV.group(1)) #print the value of the contact tactile switch
            if (int(matchV.group(1)) == 0):
                force = 20 #for SNS force
            else:
                force = 0

            toc = time.perf_counter()
            print("M43 determine contact " + str(toc - tic))
            #x, y, z, left_angle, right_angle, force = x_d, y_d, z_d,leftGrasper_Rad_d, rightGrasper_Rad_d,force

            time.sleep(ts)  # sleep this period of time -> move it here because the M280 is one of the bottlenecks in the flow

            #get time
            delT = time.time()-BeginSNS_T
            logStr = ",".join([str(x) for x in [delT,x,y,z,graspAngle,force,BufferLength,move_to_pre_grasp, move_to_grasp, grasp, lift_after_grasp, move_to_pre_release,
                move_to_release, release, lift_after_release]])
            logArray.append(logStr)

        else:
            [x_d, y_d, z_d, leftGrasper_Rad_d, rightGrasper_Rad_d] = [0, 0, 0, 0, 0]
            time.sleep(gS.timeStep)

        # ArgumentDict = {"x_gantryHead": x_d, "y_BasePos": y_d, "z_AxisBar": z_d, "x_force": 50, "y_force": 500, "z_force": 500, "leftGrasper_Rad": leftGrasper_Rad_d,
        #                 "rightGrasper_Rad": rightGrasper_Rad_d, "leftGrasperTorque": 25, "rightGrasperTorque": 25}
        #
        # # ---------step the simulation----------
        # gS.stepSim(usePositionControl=True, GUI_override=False, **
        #            ArgumentDict)  # pass argument dict to function

        gS.bulletClient.stepSimulation()

        endt_loop = time.time()
        print('LoopTime: '+str(startt_loop-endt_loop)) #print the time it takes in the loop
    if log_name == "":
        log_name = str(pathlib.PurePath.joinpath(pathlib.Path.cwd(),"Data",time.strftime("%d_%b_%Y_%H_%M_%S",time.localtime())+".txt"))
    with open(log_name,"w") as logFile:
        logFile.write("\n".join(logArray))







if __name__ == "__main__":
    ser=SetupGantry()
    pick_and_place()

