### References:
# https://kevinponce.com/blog/python/send-gcode-through-serial-to-a-3d-printer-using-python/



import serial
import time
import numpy as np

def sendCommand(ser,commandstr):
    ser.write(commandstr.encode('utf-8'))
    # readval = ser.read(20)

    # print(readval)

    while True:
        line = ser.readline()
        print(line)

        if line == b'ok\n':
            break


ser = serial.Serial('COM3', 115200, timeout=2)

#ser.open()
print(ser.is_open)
time.sleep(20)
# ser.write("G28 X0 Y0 Z0\r\n".encode('utf-8'))
# #readval = ser.read(20)
#
# #print(readval)
#
# while True:
#     line = ser.readline()
#     print(line)
#
#     if line == b'ok\n':
#         break
#sendCommand(ser,"M503\r\n")
sendCommand(ser,"G28 X0 Y0 Z0\r\n")
#sendCommand(ser,"G0 F15000 X0\r\n")
time.sleep(10)
sendCommand(ser,"G90\r\n")
print("Finished Sending G90")
sendCommand(ser,"G0 F15000\r\n")
time.sleep(1)


ts = 0.3
maxFeedRate = 150*60 #mm/min
periodT = 25 #seconds.  Done to violate maximum speed of 150 mm/sec
accel = 827.48 #mm/s for x axis
d_total = 400 #mm


time.sleep(1)
sendCommand(ser,"G0 F"+format(maxFeedRate,".2f")+ "X"+format(d_total/2,".2f")+"\r\n")
time.sleep(10)

for i in range(10000):
    startt = time.time()
    x = 0.5*(d_total)*np.sin(2*np.pi*i*ts/periodT)  + 0.5*(d_total)
    feedrate = abs(0.5*d_total*2*np.pi*np.cos(2*np.pi*i*ts/periodT)*60/periodT) #mm/min.  How fast to move the x axis to the next time point
    sendCommand(ser,"G0 F"+ format(feedrate,".2f") +" X"+format(x,".2f")+" \r\n")
    time.sleep(ts)

    sendCommand(ser, "M400\r\n")
    print("M400 fin")
    endt=time.time()
    print(endt-startt)



ser.close()
print(ser.is_open)



