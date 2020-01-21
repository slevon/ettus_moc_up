#!/usr/bin/env python
import threading
from datetime import datetime, time, date
from time import sleep
from sdr import sdr

import numpy as np
np.set_printoptions(threshold=np.inf)

import matplotlib

#matplotlib.use("Qt5Agg",warn=False, force=True)
import matplotlib.pyplot as plt
from matplotlib import rc
import random

#rc('text', usetex=True)

import sys

print("start")

def get_random_payload(type):
        '''
        :param type:
        '''

        if type == sdr.TELEGRAM_REPLY_S_SHORT:
            #data='{:08x}'.format(random.randint(0, 0xFFFFFFFF)).encode('ascii')
            #Ohne DF (5Bit) hier DF0
            data='{:08x}'.format(random.randint(0x00000000, 0x07FFFFFF)).encode('ascii')
        if type == sdr.TELEGRAM_REPLY_S_LONG:
            #data = '{:022x}'.format(random.randint(0, 0XFFFFFFFFFFFFFFFFFFFFFF)).encode('ascii')
            #Ohne DF (5Bit)hier DF18
            data = '{:022x}'.format(random.randint(0X9000000000000000000000, 0X97FFFFFFFFFFFFFFFFFFFF)).encode('ascii') #df18
            data = '{:022x}'.format(random.randint(0X9800000000000000000000, 0X98FFFFFFFFFFFFFFFFFFFF)).encode('ascii') #df23
        if type == sdr.TELEGRAM_REPLY_AC:
            data = '{:04o}'.format(random.randint(0, 0o7777)).encode('ascii')

        #TODO: Hier noch für die anderen Telegramtypen die random Payload erstellen
        return data


plt.style.use('seaborn-whitegrid')
# print(plt.style.available)
if __name__ == "__main__":
    amplitude= 1.0
    telegram_rate = 200.0  # Hz
    amount = round(telegram_rate * 2)  # 5sek duration
    init_delay = 0.1

    '''
    Setting up the SDR-Class
    #########################################################
    '''
    radio = sdr()
    ret=radio.set_channel(1090e6)
    print(ret)
    if ret[0] == "err":
        sys.exit(-2)
    radio.calibrate_gain(-40)

    '''
    Setting up the example vector
    #########################################################
    '''
    telegrams = []
    type=sdr.TELEGRAM_REPLY_AC
    amplitude=0.5
    shift=10e-6
    phase=0
    muted=False
    telegrams.append({'data':get_random_payload(type),'type':type,'amplitude':amplitude,'shift':shift,
                      'phase':phase, 'muted':muted}
                    )
    type = sdr.TELEGRAM_REPLY_S_SHORT
    amplitude = 0.3
    shift=0
    telegrams.append(
        {'data': b'7FFFFFF', 'type': type, 'amplitude': amplitude, 'shift': shift, 'phase': phase,
         'muted': muted}
        )
    #shift=150e-6
    #telegrams.append(
    #    {'data': get_random_payload(type), 'type': type, 'amplitude': amplitude, 'shift': shift, 'phase': phase,
    #     'muted': muted}
    #)
    #Main Loop running the levels
    future = 0.1
    repeat = 10
    delay = 0.01
    #Create the Field:
    for telegram in telegrams:
        radio.add_telegram(telegram['type'],telegram['data'],telegram['amplitude'],telegram['shift'],telegram['phase'])


    #plot and save the created data
    fig1, ax1 = plt.subplots()
    ax1.plot(radio.tx_samples)
    ax1.set(xlabel='samples', ylabel='Level |TX|',
            title='TX Samples SDR')
    ax1.grid()
    timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")
    fig1.savefig("{}_plotwrite.png".format(timestamp))

    radio.save_samples("{}_data.npy".format(timestamp))
    plt.show()


    #create a 2nd sdr, that has no clue waht to send
    radio2=sdr()
    radio2.load_samples("{}_data.npy".format(timestamp))

    fig2, ax2 = plt.subplots()
    ax2.plot(radio2.tx_samples)
    ax2.set(xlabel='samples', ylabel='Level |TX|',
            title='TX Samples read from file')
    ax2.grid()
    timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")
    fig2.savefig("{}_plotread.png".format(timestamp))

    plt.show()