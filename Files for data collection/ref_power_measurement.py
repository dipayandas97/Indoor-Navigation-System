import serial
import time
import csv

ser = serial.Serial('/dev/ttyUSB0', 115200)
ser.flushInput()
print('setup done')

while True:
    filename = input('Enter filename: ')
    filename = '/home/dipayan/Desktop/Indoor_Nav/Data_collected/phase_2/'+filename+'.txt'
    f = open(filename,'a')
    ser.flushInput()
    n_data = 0
    
    while n_data!=50:
        try:
            ser_bytes = ser.readline()
            data = ser_bytes.decode("utf-8")
            
            if len(data) == 10:
                t_idx = data.index('T')
                a_idx = data.index('A')
                
                t_data = data[t_idx+1:t_idx+4]
                a_data = data[a_idx+1:a_idx+4]
                print(n_data, a_data)

                #print(n_data, t_data, a_data)
                
                #f.write(t_data)
                #f.write('\t')
                f.write(a_data)
                f.write('\n')

                n_data += 1
        except:
            print("Keyboard Interrupt")
            break
    f.close()
