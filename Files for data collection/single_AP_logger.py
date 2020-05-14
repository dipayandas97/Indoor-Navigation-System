import serial
import time
import csv

ser = serial.Serial('/dev/ttyUSB0', 115200)
ser.flushInput()
print('setup done')

while True:
    filename = input('Enter Node name: ')
    filename = '/home/dipayan/Desktop/Indoor_Nav/NLOS_data/AP_4/'+filename+'.txt'
    f = open(filename,'w')
    ser.flushInput()
    n_data = 0
    
    while n_data!=100:
        try:
            ser_bytes = ser.readline()
            data = ser_bytes.decode("utf-8")
            if data[0] != '-' and len(data)!=3:
                data = '-100\n' 

            print(n_data, data)
            
            f.write(data)

            n_data += 1
            
        except:
            print("Keyboard Interrupt")
            break
    f.close()
