import numpy as np
import csv
import time
import labels

#DataCombiner -- the use is to take the multiple sensor outputs and turn them into one CSV, to make feature extraction easier. Currently only supports combination of accel, gyro data.


#Array should be: [Accel,Gyro] 
filesToCombine = ["data/accelerometer-skullcrushers-1min-gibby.csv", "data/gyroscope-skullcrushers-1min-gibby.csv"]
outputFilename = "data/Combined-skullcrushers-1min-gibby.csv"
accelData = []
with open(filesToCombine[0], "r") as f:
        reader = csv.reader(f, delimiter = ",")
        headings = next(reader)
        for row in reader:
            accelData.append(row)
gyroData = []
with open(filesToCombine[1], "r") as f:
        reader = csv.reader(f, delimiter = ",")
        headings = next(reader)
        for row in reader:
            gyroData.append(row)

#Adds the gyroscope z,y,x to the end of each row in accelData
for accelRow,gyroRow in zip(accelData,gyroData):
    if(accelRow[0]!=gyroRow[0]):
        print("Data is out of sync.")
    accelRow.append(gyroRow[2])
    accelRow.append(gyroRow[3])
    accelRow.append(gyroRow[4])

with open(outputFilename, 'w',  newline='') as f:
    writer = csv.writer(f)
    writer.writerows(accelData)
    print("Data saved to: " + outputFilename)







