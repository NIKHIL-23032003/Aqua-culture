import serial
import time
data = serial.Serial(
                    'COM18',
                    baudrate = 9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    timeout=1
                    )

def Send(a):
  data.write(str.encode(a))
  print('sent............')

def Read():
  print("reading")
  d=[]
  while True:
    Data = data.readline()
    Data = Data.decode('utf-8', 'ignore')
    # print("Raw data is ---- {}  ---".format(Data))
    Data = Data.split(',')
    print(Data)
    
    if len(Data)  == 4:
      Data[3]=Data[3].replace("\r\n","")
      

      break
  return Data

