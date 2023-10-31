import subprocess
import re
import pathlib
import pandas as pd
from datetime import datetime, timezone

def parseVideoDatalog(exifLocation,VideoDirectory,dataFileDirectory,restr,toffsets,dt_creationDate,saveFileName):
    ## Parse the datalog files
    datalogDict = {"File Name": [], "Save Time": []}
    for daF in pathlib.Path(dataFileDirectory).glob('*.txt'):
        try:
            dateTimeSave = datetime.strptime(daF.parts[-1], "%d_%b_%Y_%H_%M_%S.txt")
        except Exception as e:
            print("Couldn't process " + daF.parts[-1])
            dateTimeSave = datetime.now()
        datalogDict["File Name"].append(daF.parts[-1])
        datalogDict["Save Time"].append(dateTimeSave)
        # print(pathlib.Path.stat(daF)) #datetime.datetime.fromtimestamp(pathlib.Path.stat(daF).st_ctime).strftime("%A, %B %d, %Y %H:%M:%S")

    DataDict = {"FileName": [], "CreateDate": [], "Duration": [], "SourceImageWidth": [], "SourceImageHeight": [],
                "CameraModelName": [], "ShutterSpeed": [], "FNumber": [], "ISO": [], "ClosestDatalog": []}
    for k in pathlib.Path(VideoDirectory).glob('*.*'):
        if (re.match(".*(MOV)|.*(MP4)", str(k)) is not None):  #

            if datetime.date(datetime.fromtimestamp(pathlib.Path.stat(k).st_ctime)) == datetime.date(dt_creationDate):
                exifP = subprocess.run([exifLocation, str(k)], capture_output=True)
                MetaDataArray = []
                reMatch = re.finditer(restr, str(exifP.stdout, 'utf-8'), re.MULTILINE)
                for ma in reMatch:
                    # if count == 0:
                    #     print(",".join([k for (k,v) in ma.groupdict().items()]))
                    #     count +=1
                    valueDict = {kk: v.strip() for (kk, v) in ma.groupdict().items() if v is not None}
                    DataDict[list(valueDict.keys())[0]].append(list(valueDict.values())[0])
                    MetaDataArray.append([v.strip() for (k, v) in ma.groupdict().items() if v is not None][0])

                print(",".join(MetaDataArray))
                # check which datalog is closest in time

                dT = datetime.strptime(DataDict["CreateDate"][-1], "%Y:%m:%d %H:%M:%S.%f")

                delTvec = [abs((dT - x).total_seconds() + toffsets[DataDict["CameraModelName"][-1]]) for x in
                           datalogDict["Save Time"]]  # correct for offsets between cameras system time and PC clock time
                minTime = min(delTvec)
                minIdx = delTvec.index(minTime)
                closestDatalog = datalogDict["File Name"][minIdx]
                print("Closest Datalog " + closestDatalog)
                DataDict["ClosestDatalog"].append(closestDatalog)

    DF = pd.DataFrame.from_dict(DataDict)
    print(DF)
    DF.to_csv(saveFileName)

exifLocation = "C:\\Users\\Ravesh\\Downloads\\exiftool-12.44\\exiftool.exe" #locatiion to find the exiftool to extract metadata

VideoDirectory = "D:\\DCIM\\106_PANA\\"
dataFileDirectory = pathlib.Path.cwd()

restr = "(^File Name.*:(?P<FileName>.*))|.*(Create Date.*[\r\n]Date/Time Original\s+:?(?P<CreateDate>.*))|(Track Duration.*?:(?P<Duration>.*))|(Source Image Width.*:(?P<SourceImageWidth>.*))|(Source Image Height.*:(?P<SourceImageHeight>.*))|(Camera Model Name.*:(?P<CameraModelName>.*))|(^Exposure Time.*:(?P<ShutterSpeed>.*))|(F Number.*:(?P<FNumber>.*))|(^ISO\s*:(?P<ISO>.*))" #https://regex101.com/r/TgrMcG/1

toffsets = {"Canon EOS Rebel T7":43200, "DC-FZ80":132}#canon was -65 yet


#for sept 8th data
saveFileName = "FRR_8thSept2022_VideoInfo.csv"
dt=datetime(2022,9,8)
parseVideoDatalog(exifLocation,VideoDirectory,dataFileDirectory,restr,toffsets,dt,saveFileName)

#for sept 14th data
saveFileName = "FRR_14thSept2022_VideoInfo.csv"
dt=datetime(2022,9,14)
parseVideoDatalog(exifLocation,VideoDirectory,dataFileDirectory,restr,toffsets,dt,saveFileName)



