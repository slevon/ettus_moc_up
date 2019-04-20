#!/usr/bin/env python
import threading
from datetime import datetime, time, date
from time import sleep
from sdr import sdr

import numpy as np
np.set_printoptions(threshold=np.inf)

import matplotlib
matplotlib.use("Qt5Agg",warn=False, force=True)
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

import sys

print("start")


plt.style.use('seaborn-whitegrid')
# print(plt.style.available)
if __name__ == "__main__":
    amplitude= 1.0
    telegram_rate = 200.0  # Hz
    amount = round(telegram_rate * 5)  # 5sek duration
    init_delay = 0.1
    sample_rate = 20e6  #20MSPS


    '''
    Setting up the plots
    #########################################################
    '''
    fig = plt.figure(figsize=(9, 6))  # inches
    ax_tx_abs = fig.add_subplot(321)
    ax_tx_abs.set_title('$|TX|$')
    ax_tx_phase = fig.add_subplot(322)
    ax_tx_phase.set_title('$TX_\phi$')
    ax_rx_abs = fig.add_subplot(323)
    ax_rx_abs.set_title('$RX_{dBm}$')
    ax_rx_phase = fig.add_subplot(324)
    ax_rx_phase.set_title('$RX_\phi$ ')
    ax_rx_i = fig.add_subplot(325)
    ax_rx_i.set_title('$I-Channel_{RX}$')
    ax_rx_i.set_xlabel('$time/\mu s$')
    ax_rx_q = fig.add_subplot(326)
    ax_rx_q.set_title('$Q-Channel_{RX}$')
    ax_rx_q.set_xlabel('$time/\mu s$')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    line_RX = None

    '''
    Setting up the SDR-Class
    #########################################################
    '''
    radio = sdr()
    radio.set_channel(1090e6,sample_rate=sample_rate)


    '''
    Setting up the example vector
    #########################################################
    '''

    data=b'7777'

    tx_vector = np.array([0,0,0,0], dtype=np.complex64) # Some samples for time to start
    a = amplitude

    '''
    MODE A/C Reply consitst of 2 to 15 pulses:
    F1 und F2 Pulse: Width 5us Distance 20.3us
    data pulses: width 0.45us distance 1us
    '''
    symbol_time = 0.45e-6  # 0.45us
    pause_time = 1e-6  # 1us
    samples_per_symbol = symbol_time * sample_rate
    samples_per_pause = pause_time * sample_rate

    one   = np.repeat(np.array([a + 0j], dtype=np.complex64), samples_per_symbol)  # 4.5us
    zero  = np.repeat(np.array([0 + 0j], dtype=np.complex64), samples_per_symbol)  # 4.5us
    pause = np.repeat(np.array([0 + 0j], dtype=np.complex64), samples_per_pause)   # 10us
    bit = {}
    bit[0] = np.append(zero, pause)
    bit[1] = np.append(one, pause)
    a = int(data[0])
    b = int(data[1])
    c = int(data[2])
    d = int(data[3])

    tx_vector = np.concatenate((tx_vector,bit[1])) #F1 Pulse
    tx_vector = np.concatenate((tx_vector, bit[c & 0b001]))  # C1 Pulse
    tx_vector = np.concatenate((tx_vector, bit[a & 0b001]))  # A1 Pulse
    tx_vector = np.concatenate((tx_vector, bit[bool(c & 0b010)]))  # C2 Pulse
    tx_vector = np.concatenate((tx_vector, bit[bool(a & 0b010)]))  # A2 Pulse
    tx_vector = np.concatenate((tx_vector, bit[bool(c & 0b100)]))  # C4 Pulse
    tx_vector = np.concatenate((tx_vector, bit[bool(a & 0b100)]))  # A4 Pulse
    tx_vector = np.concatenate((tx_vector, bit[0]))  # X Pulse
    tx_vector = np.concatenate((tx_vector, bit[b & 0b001]))  # B1 Pulse
    tx_vector = np.concatenate((tx_vector, bit[d & 0b001]))  # D1 Pulse
    tx_vector = np.concatenate((tx_vector, bit[bool(b & 0b010)]))  # B2 Pulse
    tx_vector = np.concatenate((tx_vector, bit[bool(d & 0b010)]))  # D2 Pulse
    tx_vector = np.concatenate((tx_vector, bit[bool(b & 0b100)]))  # B4 Pulse
    tx_vector = np.concatenate((tx_vector, bit[bool(d & 0b100)]))  # D4 Pulse
    tx_vector = np.concatenate((tx_vector, bit[1]))  # F2 Pulse
    tx_vector = np.concatenate((tx_vector, bit[0]))  # pause
    tx_vector = np.concatenate((tx_vector, bit[0]))  # pause
    tx_vector = np.concatenate((tx_vector, bit[0]))  # pause
    tx_vector = np.concatenate((tx_vector, bit[0]))  # pause
    tx_vector = np.concatenate((tx_vector, bit[0]))  # pause

    #Main Loop running the levels
    for level in np.arange(-60, -91, -1.0):
        level_start = datetime.utcnow()
        level_total_amount = 0
        print("{}\tSending {} Telegrams {}dBm with {}vcts/s: ".format(datetime.utcnow(), amount, level, telegram_rate)),

        thread_tx = threading.Thread(target=radio.send_telegram, args=(tx_vector,level, init_delay, amount, 1 / telegram_rate))
        thread_rx = threading.Thread(target=radio.receive_telegram)
        thread_tx.start()
        sleep(0.001)
        thread_rx.start()
        thread_rx.join()    #RX finished very fast, only the first transmission of the loop is received
        plt.suptitle(
            '$\\textbf{{ {} }}$,  {}dBm, {}vcts/s'.format("Test", level,
                                                            telegram_rate),
            fontsize=18)

        rx_vector_db = radio.rx_samples_db
        rx_vector = radio.rx_samples
        #First iteration creates the plots after that, the exiting plots are updated   
        if not line_RX:
            time_line = 1e6 * np.arange(0, (len(rx_vector_db)) / radio.sample_rate, 1 / radio.sample_rate)[
                              :len(rx_vector_db)]
            time_line_tx = 1e6 * np.arange(0, (len(tx_vector)) / radio.sample_rate, 1 / radio.sample_rate)[
                                 :len(tx_vector)]
            line_I, = ax_rx_i.plot(time_line, rx_vector.real, 'm-')
            line_Q, = ax_rx_q.plot(time_line, rx_vector.imag, 'b-')
            line_RX, = ax_rx_abs.plot(time_line, rx_vector_db, 'g-')

            line_RX_phi, = ax_rx_phase.plot(time_line, np.angle(rx_vector), 'r-')
            line_TX, = ax_tx_abs.plot(time_line_tx, np.absolute(tx_vector), 'g-')
            line_TX_phi, = ax_tx_phase.plot(time_line_tx, np.angle(tx_vector), 'r-')

        else:
            line_I.set_ydata( rx_vector.real)
            line_Q.set_ydata(rx_vector.imag)
            line_RX.set_ydata(rx_vector_db)
            line_RX_phi.set_ydata( np.angle(rx_vector))
            line_TX.set_ydata(np.absolute(tx_vector))
            line_TX_phi.set_ydata(np.angle(tx_vector))

        #Wait for the transmission to end
        while thread_tx.is_alive():
           plt.pause(0.0000001)


        level_total_amount += amount
        rate = level_total_amount / ((datetime.utcnow() - level_start).total_seconds() - init_delay)
        print("\tActual: {:.2f} pkt/s".format(rate))

    plt.show()
