import numpy as np



# constant
LOW = 0
HIGH = 1

enable = 0
gpio_pin = np.arange(1,41)
gpio_pin_state = np.zeros((40))
gpio_pin = np.column_stack((gpio_pin,gpio_pin_state))

def BOARD(self):    
    return "GPIO as board"

def setmode(a):
    if a != "GPIO as board":
        print("Plase set GPIO as Board")
    else:
        enable = 1

def setup(a,b):
    return 0

def OUT():
    return 0

def output(a, b):
    gpio_pin[a-1][1] = b
    f = open("./datafile/control.txt", 'w')
    for i in range(len(gpio_pin)):
        data = str(gpio_pin[i]).split('.')
        a = data[0].strip('[')
        b = data[1].strip(' ')
        f.write(a+' '+b+'\n')
    f.close()
    
def cleanup():
    f = open("./datafile/control.txt", 'w')
    for i in range(len(gpio_pin)):
        data = str(gpio_pin[i]).split('.')
        a = data[0].strip('[')
        f.write(a+' '+'0'+'\n')
    f.close()

class pin:
    def __init__(self, name):
        self.name = name

class PWM:  
    def __init__(self, pin, cycle ):
        self.pin = pin-1
        self.cycle = cycle
        self.enable = 0
        self.duty = 0
    def start(self, duty):
        self.enable = 1
        if duty >= 0 and duty <= 100:
            self.duty = duty
            gpio_pin[self.pin][1] = round(duty*0.01,4)
            f = open("./datafile/control.txt", 'w')
            for i in range(len(gpio_pin)):
                data = gpio_pin[i].tolist()
                a = int(data[0])
                b = data[1]
                f.write(str(a)+'0 '+str(b)+'0\n')
            f.close()
        
    def changeDutyCycle(self, duty):
        if self.enable == 1:
            if duty >= 0 and duty <= 100:
                self.duty = duty
                gpio_pin[self.pin][1] = round(duty*0.01,4)
                f = open("./datafile/control.txt", 'w')
                for i in range(len(gpio_pin)):
                    data = gpio_pin[i].tolist()
                    a = int(data[0])
                    b = data[1]
                    out = str(a)+' '+str(b)+'\n'
                    f.write(out)
                f.close()
          
       
    

    
    
    
    




    