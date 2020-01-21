#!/usr/bin/env python

'''
 SDR-Class that abstacts the real SDR to the testdatagenerator
'''
import logging
from datetime import datetime, date

import math
from time import sleep

import uhd
import numpy as np
import uhd.libpyuhd as lib
import threading

np.set_printoptions(threshold=np.inf)

class LogFormatter(logging.Formatter):
    """Log formatter which prints the timestamp with fractional seconds"""

    @staticmethod
    def pp_now():
        """Returns a formatted string containing the time of day"""
        now = datetime.now()
        return "{:%Y-%m-%d %H:%M}:{:05.2f}".format(now, now.second + now.microsecond / 1e6)
        # return "{:%H:%M:%S}".format(now)

    def formatTime(self, record, datefmt=None):
        converter = self.converter(record.created)
        if datefmt:
            formatted_date = converter.strftime(datefmt)
        else:
            formatted_date = LogFormatter.pp_now()
        return formatted_date


class sdr:
    TELEGRAM_REPLY_AC = 1
    TELEGRAM_REPLY_S_SHORT = 2
    TELEGRAM_REPLY_S_LONG = 3
    TELEGRAM_CALL_A = 11
    TELEGRAM_CALL_S_ALL = 12
    #Todo: TELEGRAM_CALL_S_ROLL = 13

    def __init__(self):
        """
            Constructor of the SDR-Class
        """
        self.usrp = None ## USRP Object

        self.tx_level=None  ##T he HF-Level for sending calibrated via self.calibrate_gain
        self._gain_offset = None ## Set using self.calibrate_gain
        self.tx_max_output_dbm = None ## \see set_channel
        self.tx_data=""     ## Current Payload for send
        self.tx_telegram=None ## Current TelegramType
        self.number_of_telegrams = 0    ## Holds the number Telegrams in the current buffer

        self.threads = []       ## Todo: Still used?
        self.quit_event = False ## Todo: Still used?


        self.sample_rate = 20e6 ## MSPS
        self.frequency = -1     ## Must be either 1090e6 or 1030e6 \see self.set_channel
        self.tx_channel = -1    ## Number of the SDR-Channel: Default: 0 TX/RX1
        self._tx_streamer = None  ## Holds the UHD-Streamer to the SDR
        self.tx_samples = np.array([])       ## Holds the samples to be send for each iteration
        self.rx_samples = np.array([])      ## Holds the samples of the received signal
        self.transmitting = False  ## The current SDR state #TODO: Still used?
        self.current_time_spec = False ## The current timespec for the next transmission, FALSE if not active


        ''' Logger 
            The Logger is set up for console output
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        console = logging.StreamHandler()
        self.logger.addHandler(console)
        logfile = logging.FileHandler('sdr.log')
        logfile.setLevel(logging.DEBUG)
        self.logger.addHandler(logfile)
        formatter = LogFormatter(fmt='[%(asctime)s] [%(levelname)s] %(message)s')
        console.setFormatter(formatter)
        logfile.setFormatter(formatter)

    def connect(self):
        """
            Connects to the SDR and sets the time.
        :return: False if the connection failed, else the USRP definition string
        """
        try:
            self.usrp = uhd.usrp.MultiUSRP()
        except:
            self.usrp = None
            return False

        """Reset the SDR internal clock"""
        self.usrp.set_time_now(lib.types.time_spec(0))
        self.logger.info("Using Device: %s", self.usrp.get_pp_string())


        return self.usrp.get_pp_string()

    def set_channel(self, frequency, tx_channel=0):
        """
            Sets the SDR to the channel 1030e6 or 1090e6.
            If the SDR is not connected it will try to connect it.
            It also sets the analog Bandwidth and the sample rate.
            At last it creates a streamer object to which the data can be sent
        :param frequency: Allowed values 1090e6 or 1030e6
        :param tx_channel: The tx-Channel of the SDR (default =0)
        :return: returns a tuple of "succ" and des pp_string of the sdr in case of success
        """
        if not self.usrp:
            if not self.connect():
                return "err","No SDR connected"

        if frequency != 1090e6 and frequency != 1030e6:
            return "err", "Wrong Frequnecy selected"

        self.frequency = frequency
        self.tx_channel = tx_channel
        if self.frequency == 1090e6:
            self.sample_rate = 20e6  #
            self.usrp.set_tx_bandwidth(1.3e6)
        else:
            self.sample_rate = 20e6  #  TODO: Define samplerate for 1030 DPSK
            self.usrp.set_tx_bandwidth(4e6)

        self.usrp.set_tx_rate(self.sample_rate)
        print("BANDWITH: ",self.usrp.get_tx_bandwidth())
        self.usrp.set_tx_freq(lib.types.tune_request(self.frequency), )

        tx_cpu = "fc32"  # floatng complex 32bit
        tx_otw = "sc16"  # int complex 16bit
        st_args = uhd.usrp.StreamArgs(tx_cpu, tx_otw)
        st_args.channels = [tx_channel]
        self._tx_streamer = self.usrp.get_tx_stream(st_args)

        return "succ", "Using Device:" + self.usrp.get_pp_string()


    def calibrate_gain(self, gain_offset):
        '''
        This calibartion must be changed for each HF-Environment.
        It is used to calibrate the SDR and HF-Lane to create valid levels according to the "level" parameter

        :param gain_offset: The Offest in dBm to achive a valid level
        :return:
        '''
        self._gain_offset=gain_offset

    def clear_tx_samples(self):
        '''
        Clears the buffer for tx
        :return:
        '''
        self.tx_samples = np.array([], dtype=np.complex64)
        self.tx_telegram = None
        self.tx_data = ""
        self.number_of_telegrams = 0

    def set_telegram(self, telegram_type, data, amplitude=1.0):
        '''
        Shortcut for_
        clear_txsamples()
        add_telegram()
        :param telegram_type:
        :param data:
        :param amplitude:
        :return:
        '''
        self.clear_tx_samples()
        self.add_telegram(telegram_type,data,amplitude)


    def add_telegram(self, telegram_type, data, amplitude=1.0, shift=0.0, vector_phase=0):
        '''
        Adds a new Telegram to the output buffer
        :param telegram_type: Type of Telegram
        :param data: Payload (if applicable) MODE-A ->oct, MODE-S-RPLY=hex
        :param amplitude: Value for the vector amplitude
        :param shift: value in seconds to delay the start of the vector
        :param vector_phase: The phase of the vector in rad
        :return:
        '''
        if self.frequency == 1090e6:
            if telegram_type > 10:
                raise ValueError('Telegram Type not valid for current frequency {}'.format(self.frequency))
        if self.frequency == 1030e6:
            if telegram_type < 10:
                raise ValueError('Telegram Type not valid for current frequency {}'.format(self.frequency))


        self.number_of_telegrams+=1

        new_vector = np.array([0,0,0,0], dtype=np.complex64) # Some samples for time to start

        #delay
        samples_delay = shift * self.sample_rate
        #print("~~~Delay {} Adding {}".format(shift,samples_delay))
        delay_vector = np.repeat(np.array([0 + 0j], dtype=np.complex64), samples_delay)
        new_vector = np.concatenate((new_vector,delay_vector))

        a = amplitude

        self.tx_telegram =  telegram_type
        if len(data) > 8:
            strdata=str(data[:8])[2:-1] + '...'
        else:
            strdata=str(data)[2:-1]
        self.tx_data += strdata + ", "
        if telegram_type == self.TELEGRAM_REPLY_AC:
            '''
            MODE A/C Reply consitst of 2 to 15 pulses:
            F1 und F2 Pulse: Width 5us Distance 20.3us
            data pulses: width 0.45us distance 1us
            '''
            symbol_time = 0.45e-6  # 0.45us
            pause_time = 1e-6  # 1us
            samples_per_symbol = symbol_time * self.sample_rate
            samples_per_pause = pause_time * self.sample_rate

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

            new_vector = np.concatenate((new_vector,bit[1])) #F1 Pulse
            new_vector = np.concatenate((new_vector, bit[c & 0b001]))  # C1 Pulse
            new_vector = np.concatenate((new_vector, bit[a & 0b001]))  # A1 Pulse
            new_vector = np.concatenate((new_vector, bit[bool(c & 0b010)]))  # C2 Pulse
            new_vector = np.concatenate((new_vector, bit[bool(a & 0b010)]))  # A2 Pulse
            new_vector = np.concatenate((new_vector, bit[bool(c & 0b100)]))  # C4 Pulse
            new_vector = np.concatenate((new_vector, bit[bool(a & 0b100)]))  # A4 Pulse
            new_vector = np.concatenate((new_vector, bit[0]))  # X Pulse
            new_vector = np.concatenate((new_vector, bit[b & 0b001]))  # B1 Pulse
            new_vector = np.concatenate((new_vector, bit[d & 0b001]))  # D1 Pulse
            new_vector = np.concatenate((new_vector, bit[bool(b & 0b010)]))  # B2 Pulse
            new_vector = np.concatenate((new_vector, bit[bool(d & 0b010)]))  # D2 Pulse
            new_vector = np.concatenate((new_vector, bit[bool(b & 0b100)]))  # B4 Pulse
            new_vector = np.concatenate((new_vector, bit[bool(d & 0b100)]))  # D4 Pulse
            new_vector = np.concatenate((new_vector, bit[1]))  # F2 Pulse
            new_vector = np.concatenate((new_vector, bit[0]))  # pause
            new_vector = np.concatenate((new_vector, bit[0]))  # pause
            new_vector = np.concatenate((new_vector, bit[0]))  # pause
            new_vector = np.concatenate((new_vector, bit[0]))  # pause
            new_vector = np.concatenate((new_vector, bit[0]))  # pause

        if telegram_type == self.TELEGRAM_REPLY_S_SHORT or telegram_type == self.TELEGRAM_REPLY_S_LONG:
            '''
            MODE-S short reply consists of:
            
            8us: Preamble:
                1. Pulse: 0,5us 
                2. Pulse: 0,5us  + 1 us
                3. Pulse: 0,5us  + 3.5 us
                4. Pulse: 0,5us  + 4.5 us
            56us Datablock:
            '''
            symbol_time = 0.5e-6  # 0.5us
            samples_per_symbol = symbol_time * self.sample_rate
            pause_time = 1e-6  # 2us Preamble between 2.Pluse and 3. Pulse
            samples_per_pause = pause_time * self.sample_rate

            high = np.repeat(np.array([a + 0j], dtype=np.complex64), samples_per_symbol)  # 0.5us
            low = np.repeat(np.array([0 + 0j], dtype=np.complex64), samples_per_symbol)  # 0.5us
            pause = np.repeat(np.array([0 + 0j], dtype=np.complex64), samples_per_pause)  # 1us
            bit = {}
            bit[0] = np.append(low, high)
            bit[1] = np.append(high, low)
            
            new_vector = np.concatenate((new_vector, bit[1]))  # # P1 Pulse
            new_vector = np.concatenate((new_vector, bit[1]))  # # P2 Pulse
            new_vector = np.concatenate((new_vector, pause))  # preamble pause
            new_vector = np.concatenate((new_vector, low))    # preamble pause/2
            new_vector = np.concatenate((new_vector, bit[1]))  # preamble P3
            new_vector = np.concatenate((new_vector, bit[1]))  # preamble P4
            new_vector = np.concatenate((new_vector, pause))  # preamble pause
            new_vector = np.concatenate((new_vector, pause))  # preamble pause
            new_vector = np.concatenate((new_vector, low))  # preamble pause/2

            #format
            # if telegram_type == self.TELEGRAM_REPLY_S_SHORT:
            #     format = sdr.bin2np("00000") # DF00
            # else:
            #     format = sdr.bin2np("10010") # DE18
            # for i in format:
            #     new_vector = np.concatenate((new_vector, bit[int(i)]))  # Data

            if telegram_type == self.TELEGRAM_REPLY_S_SHORT:
                msg = sdr.hex2bin(data,32)
            else:
                msg = sdr.hex2bin(data,32+56)

            for i in msg:
                new_vector = np.concatenate((new_vector, bit[int(i)]))  # Data
            crc_field=data

            if telegram_type == self.TELEGRAM_REPLY_S_SHORT:
                crc = sdr.crc(crc_field, 32, True)
            else:
                crc = sdr.crc(crc_field, 32+56, True)  # Long

            for i in crc:
                new_vector = np.concatenate((new_vector, bit[int(i)]))  # crc
            new_vector = np.concatenate((new_vector, pause))  # time to rest
            new_vector = np.concatenate((new_vector, pause))  # time to rest
            new_vector = np.concatenate((new_vector, pause))  # time to rest
            new_vector = np.concatenate((new_vector, pause))  # time to rest

        if telegram_type == self.TELEGRAM_CALL_A:
            '''
            MODE-A Call (Uplink):

            8us: Preamble:
                1. Pulse1: 0,8us 
                2. Pulse3: 0,8us  + 8 us
            '''
            symbol_time = 0.8e-6  # us
            samples_per_symbol = symbol_time * self.sample_rate

            pause_time_p2 = 2e-6 - symbol_time  # us Preamble between Pulses
            samples_per_pause_p2 = pause_time_p2 * self.sample_rate

            pause_time = 8e-6 - pause_time_p2 - 2 * symbol_time # us  between Pulses
            samples_per_pause = pause_time * self.sample_rate

            high = np.repeat(np.array([a + 0j], dtype=np.complex64), samples_per_symbol)  # Pulse
            pause = np.repeat(np.array([0 + 0j], dtype=np.complex64), samples_per_pause)  # Pause
            pause_p2 = np.repeat(np.array([0 + 0j], dtype=np.complex64), samples_per_pause_p2)  # Pause from p1 to p2

            new_vector = np.concatenate(( new_vector, pause))  # time to start
            new_vector = np.concatenate((new_vector, high))  # # P1 Pulse
            new_vector = np.concatenate((new_vector, pause_p2))  # # P2 Delay
            new_vector = np.concatenate((new_vector, high*0.25))  # # P2
            new_vector = np.concatenate((new_vector, pause))  # # Pause
            new_vector = np.concatenate((new_vector, high))  # # P3 Pulse
            new_vector = np.concatenate((new_vector, pause))  # # P3 Pulse
            new_vector = np.concatenate((new_vector, pause))  # # Time to rest
            new_vector = np.concatenate((new_vector, pause))  # # Time to rest


        if telegram_type == self.TELEGRAM_CALL_S_ALL:
            '''
            MODE-S Call All-Call (UF11) DPSK

            8us: Preamble:
                1. Pulse: 0,8us 
                2. Pulse: 0,8us  + 1.2 us
                3. Pulse: 1.25us  + 3.5 us
                4. DATA block DBPSK
            56us Datablock:
            II code 14 and SI 14 R&D reserved codes
            
            '''
            symbol_time = 0.25e-6
            samples_per_symbol = symbol_time * self.sample_rate
            samples_per_symbol_pulse = symbol_time * self.sample_rate
            pause_time = 1.2e-6
            samples_per_pause = pause_time * self.sample_rate

            inphase = np.repeat(np.array([a + 0j], dtype=np.complex64), samples_per_symbol)  # 0.25us
            outphase = np.repeat(np.array([-1*a + 0j], dtype=np.complex64), samples_per_symbol)  # 0.25us
            pause = np.repeat(np.array([0 + 0j], dtype=np.complex64), samples_per_pause)  # 1.2us
            pulse = np.repeat(np.array([a + 0j], dtype=np.complex64), samples_per_symbol_pulse)  # 0.8us
            bit = {}
            bit[True] = inphase
            bit[False] = outphase

            new_vector = np.concatenate((new_vector, pulse))  # # P1 Pulse
            new_vector = np.concatenate((new_vector, pause))  # preamble pause
            new_vector = np.concatenate((new_vector, pulse))  # # P2 Pulse
            new_vector = np.concatenate((new_vector, pause))  # preamble pause
            new_vector = np.concatenate((new_vector, inphase))  # sync
            new_vector = np.concatenate((new_vector, inphase))  # sync
            new_vector = np.concatenate((new_vector, inphase))  # sync
            new_vector = np.concatenate((new_vector, inphase))  # sync
            new_vector = np.concatenate((new_vector, inphase))  # sync
            new_vector = np.concatenate((new_vector, outphase))  # sync revers
            new_vector = np.concatenate((new_vector, outphase))  # sync revers
            #UF11    PR    II=14  CL=0      16Bit Spare + 24Bit '1'
            msg = '01011'+'0101'+'1110'+'000'+sdr.hex2bin("00FFFFFF",40)
            lastChip=False

            for i in msg:
                if i == '1':
                    lastChip = not lastChip
                new_vector = np.concatenate((new_vector, bit[lastChip]))  # Data

            new_vector = np.concatenate((new_vector, bit[lastChip]))  # secure space
            new_vector = np.concatenate((new_vector, bit[lastChip]))  # secure space
            new_vector = np.concatenate((new_vector, pause))  # time to rest
            new_vector = np.concatenate((new_vector, pause))  # time to rest
            new_vector = np.concatenate((new_vector, pause))  # time to rest
            new_vector = np.concatenate((new_vector, pause))  # time to rest
            new_vector = np.concatenate((new_vector, pause))  # time to rest

        #After the new vector is created:
        # We can add them to the current tx buffer
        if self.tx_samples.shape[0] >  new_vector.shape[0]:
            new_vector.resize(self.tx_samples.shape)
        else:
            self.tx_samples.resize(new_vector.shape)

        #change Phase if requested:
        if vector_phase != 0:
            print("Phase change {}".format(vector_phase))
            r,theta = self.z2polar(new_vector)
            theta += vector_phase
            new_vector = self.polar2z(r,theta)

        #Add to ouput puffer
        self.tx_samples = self.tx_samples + new_vector

    @staticmethod
    def polar2z(r, theta):
        return r * np.exp(1j * theta)

    @staticmethod
    def z2polar(z):
        return (abs(z), np.angle(z))

    def set_tx_level(self,level):
        '''
        Sets the tx Level
        :return:

         The TX level needs to be calibated. This ist done using the SBP3 and an Splitter:
         The dB value for 0dB Gain is: -69.7341127997dBm
        '''
        # Todo: Raise Value errors for wrong levels:
        if level != self.tx_level:  # only on changes
            self.tx_level = level
            self.usrp.set_tx_gain(self.tx_level + 60 + 50 + self._gain_offset)

    def _get_leveled_tx_samples(self):
        '''
        Levels the samples to the output -1 dBm to avoid clipping
        :return:
        '''
        if np.max(np.abs(self.tx_samples)) > 1:
            raise ValueError(
                'Telegram has no vaild magnitude. Must be <= 1  but is {}'.format(np.max(np.abs(self.tx_samples))))

        return self.tx_samples * 0.7943282  # Avoid Clipping of the sender, -1dBm
        # #pow(10, (self.tx_level - self.tx_max_output_dbm) / 10.0) * self.tx_samples

    def save_samples(self,filename):
        """
            Saves the current output buffer to a given filename
        :param filename:
        """
        np.save(filename,self.tx_samples)

    def load_samples(self,filename):
        '''
        Lodas the np sampels from a file to send them:
        :param filename: the np smaples file
        :return:
        '''
        self.tx_samples=np.load(filename)

    def receive_async_tx(self,timeout):
        '''
         Receive and process the asynchronous TX messages
        :param: timeout
        :return: Returns the number of successfull transmissions (0 or 1)
        '''

        success = 0
        async_metadata = uhd.types.TXAsyncMetadata()
        # Receive the async metadata
        while not self._tx_streamer.recv_async_msg(async_metadata, timeout * 1.9):
            print(".", end='')

        # Handle the error codes
        if async_metadata.event_code == uhd.types.TXMetadataEventCode.burst_ack:
            #print("#" ,end='')
            success=1
        elif ((async_metadata.event_code == uhd.types.TXMetadataEventCode.underflow) or
              (async_metadata.event_code == uhd.types.TXMetadataEventCode.underflow_in_packet)):
            print("UDR", end='')
            self.logger.error("UDR")
            success = 0
        elif ((async_metadata.event_code == uhd.types.TXMetadataEventCode.seq_error) or
              (async_metadata.event_code == uhd.types.TXMetadataEventCode.seq_error_in_packet)):
            print("SEQ", end='')
            self.logger.error("SEQ")
            success=0
        elif (async_metadata.event_code == uhd.types.TXMetadataEventCode.time_error):

            # If it is more important to send the packets with a given rate, This error can be irgnored by commenting out this block
            print("LC", end='')
            self.logger.error("LC")
            # metadata.time_spec = uhd.types.TimeSpec(self.usrp.get_time_now().get_real_secs())
            success = 0
        else:
            self.logger.warning("Unexpected event on async recv (%s), continuing.",
                                async_metadata.event_code)
            success = 0

        return success


    def send(self,level,future=0, blocking=False):
            '''
            Sends a single Telegram of the  current active Vector
            :param level:
            :param future:
            :param blocking: If True, this function returns only after the packet was sent
            :return:
            '''

            self.set_tx_level(level)
            leveled_tx_samples = self._get_leveled_tx_samples()
            num_of_samples = len(leveled_tx_samples)
            stream_send_timeout = num_of_samples / self.sample_rate * 0.8
            #np.savetxt("tx_vector.csv", leveled_tx_samples)

            metadata = uhd.types.TXMetadata()
            if future != 0:
                metadata.has_time_spec = True
                self.current_time_spec = uhd.types.TimeSpec(self.usrp.get_time_now().get_real_secs() + future)
                # print("Time Spec {}".format(self.current_time_spec.get_real_secs()))
                metadata.time_spec = self.current_time_spec
            else:
                self.current_time_spec = False
                metadata.has_time_spec = False

            self._tx_streamer.send(leveled_tx_samples, metadata, stream_send_timeout)
            '''
            Receive if successful:
            '''
            if blocking:
                return self.receive_async_tx()


    def send_repeat(self, level, future, repeat, delay):
        '''
        Sends out the created vector
        :param level:
        :param future:
        :param repeat:
        :param delay: val
        :return:
        '''
        self.set_tx_level(level)

        leveled_tx_samples=self._get_leveled_tx_samples()
        num_of_samples=len(leveled_tx_samples)
        stream_send_timeout=num_of_samples/self.sample_rate*0.9
        #stream_send_timeout=5
        print("-Check Numof Sampls", float(num_of_samples), self._tx_streamer.get_max_num_samps(), stream_send_timeout)
        ##Debug
        #np.savetxt("tx_vector.csv", np.abs(leveled_tx_samples))

        packets_per_vector = math.ceil(float(num_of_samples) / self._tx_streamer.get_max_num_samps())

        #with open(datetime.utcnow().isoformat()+".vct", 'a') as f:
        #    f.write(leveled_tx_samples)

        '''
        The TX level needs to be calibated. This ist done using the SBP3 and an Splitter:
        The dB value for 0dB Gain is: -69.7341127997dBm
        '''
        ''' calib_value = -69.7341127997
        tx_gain = 10

        self.tx_max_output_dbm = calib_value + tx_gain
        self.usrp.set_tx_gain(tx_gain)  # This will create -59734111dBm-Fullscale
        
        '''
        metadata = uhd.types.TXMetadata()
        # Send a mini EOB packet
        #metadata.end_of_burst = True  #After each packet: Set EOB for the SDR, so that there is noch Buffer underrun "UUUUU"

        #print(np.abs(leveled_tx_samples))
        if future != 0:
            metadata.has_time_spec = True
            #V1
            first_timespec = self.usrp.get_time_now()
            self.current_time_spec = uhd.types.TimeSpec(int(first_timespec.get_real_secs()), first_timespec.get_frac_secs()+ future)
            #V2
            #first_timespec = self.usrp.get_time_now() + uhd.types.TimeSpec(future)
            #self.current_time_spec = first_timespec
            self.logger.debug("Start: Time Spec {:.3f}".format(self.current_time_spec.get_real_secs()))
            metadata.time_spec = self.current_time_spec
        else:
            self.current_time_spec = False
            metadata.has_time_spec = False

        metadata.end_of_burst = True

        #V1
        startdate_full=int(metadata.time_spec.get_real_secs()) # UHD binding, does not support get_full_secs()
        startdate_frac=metadata.time_spec.get_frac_secs()
        #V2
        #start_time_spec=metadata.time_spec

        err_send_samples = 0
        err_send_async = 0
        try:
            start_loop_time = datetime.utcnow()
            for i in range(repeat):
                start=datetime.utcnow()
                #print("Begin",start)                                        #The timeout:
                sampels_sent = self._tx_streamer.send(leveled_tx_samples, metadata, stream_send_timeout)
                #print("Send took",(start-datetime.utcnow()).total_seconds(),sampels_sent)
                #Prepare next send;
                metadata = uhd.types.TXMetadata()
                metadata.end_of_burst = True
                metadata.has_time_spec = True
                #V1
                frac, full = math.modf(((i + 1) * delay))
                metadata.time_spec = uhd.types.TimeSpec(startdate_full + int(full), startdate_frac + frac)
                #V2
                #next_delay=(i + 1) * delay
                #metadata.time_spec = start_time_spec + uhd.types.TimeSpec(next_delay)
                #print("Time Spec {:.3f}".format(metadata.time_spec.get_real_secs()))
                if sampels_sent != num_of_samples:
                    print("Error send: {} {} {}".format(i,sampels_sent, num_of_samples))
                    err_send_samples += 1
                else:
                    success_now = 0
                    #First Vector
                    if i == 0:
                        success_now = self.receive_async_tx(delay + future)
                    else:
                        start = datetime.utcnow()
                        success_now = self.receive_async_tx(delay)
                        #print("REceive async  took", (start - datetime.utcnow()).total_seconds(), sampels_sent)

                    if success_now == 0:
                        err_send_async += 1
                        #self.logger.error("ERR:")
                        #curr_sec=self.usrp.get_time_now().get_real_secs()
                        #self.logger.warning("WARN: {} : {}\t{} - {} - {}\t{}".format(curr_sec -full -frac, full,frac,curr_sec, int(metadata.time_spec.get_real_secs()),metadata.time_spec.get_frac_secs(),) )
                        self.logger.error("CNT: {}\t ERR: {} Pkts/Vct: {}".format(i,err_send_async,packets_per_vector))
        except:
            self.logger.error("error in transmit: %s")
            print("SDR ERROR Runtime")
            raise Exception("SDR Error")

        self.last_tx__success_count=(repeat - err_send_samples - math.ceil(float(err_send_async)/packets_per_vector) )
        print("\nTook:",(datetime.utcnow() - start_loop_time).total_seconds(),self.last_tx__success_count , err_send_async, packets_per_vector, err_send_samples ,end="")
        return  self.last_tx__success_count

    def receive_telegram(self):
        usrp = self.usrp
        # Set the USRP rate, freq, and gain
        usrp.set_rx_rate(self.sample_rate, self.tx_channel)
        usrp.set_rx_freq(uhd.types.TuneRequest(self.frequency), self.tx_channel)
        #usrp.set_rx_gain( (130+self.tx_level), self.tx_channel)
        usrp.set_rx_gain(50, self.tx_channel)
        #usrp.set_rx_gain(42.5, self.tx_channel)

        # Create the buffer to recv samples
        num_samps = int(len(self.tx_samples)+2000)
        # self.rx_samples = np.empty((1, num_samps), dtype=np.complex64)
        self.rx_samples = np.array([], dtype=np.complex64)

        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [self.tx_channel]

        metadata = uhd.types.RXMetadata()
        streamer = usrp.get_rx_stream(st_args)
        #buffer_samps = streamer.get_max_num_samps()
        # recv_buffer = np.zeros((1, buffer_samps), dtype=np.complex64)
        recv_buffer = np.zeros((1, num_samps), dtype=np.complex64)

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        while not self.current_time_spec:
            sleep(0.001) #waiting for Timespec to be set:

        stream_cmd.stream_now = False
        stream_cmd.time_spec = uhd.types.TimeSpec(self.current_time_spec.get_real_secs())
        print("RX Time Spec {}".format(stream_cmd.time_spec.get_real_secs()))
        streamer.issue_stream_cmd(stream_cmd)
        # Receive the samples
        self.rx_samples = np.array([], dtype=np.complex64)

        recv_samps = 0
        # while self.usrp.get_time_now().get_real_secs() < self.current_time_spec:
        #    pass
        while recv_samps < num_samps:
            #print(" REceived SAMPL {}".format(recv_samps)),
            samps = streamer.recv(recv_buffer, metadata)
            # self.rx_samples = np.append(self.rx_samples,recv_buffer[samps:])
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none and metadata.error_code != uhd.types.RXMetadataErrorCode.timeout:
                print(metadata.strerror())
                pass
            if samps:
                self.rx_samples = recv_buffer[0, :samps]
                recv_samps += samps
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stream_cmd)
        #Convert data into dBm
        try:
            with np.errstate(divide='ignore'):
                self.rx_samples_db = 10 * np.log10(10* (np.power(np.real(self.rx_samples),2)+np.power(np.imag(self.rx_samples),2))) - self.usrp.get_rx_gain() + self._gain_offset
        except RuntimeWarning:
            pass  #Divied by zero in log10

        #print("Received {}".format(recv_samps))

    def type2str(self):
        '''
        Prints the current Telegram Type as String
        :return:
        '''
        if self.tx_telegram == self.TELEGRAM_REPLY_AC :
            return "Mode A/C Reply"
        if self.tx_telegram == self.TELEGRAM_REPLY_S_SHORT:
            return "Mode-S short Reply"
        if self.tx_telegram == self.TELEGRAM_REPLY_S_LONG:
            return "Mdoe-S long Reply"
        if self.tx_telegram == self.TELEGRAM_CALL_A:
            return "Mode A Call"
        if self.tx_telegram == self.TELEGRAM_CALL_S_ALL:
            return "Mode-S All-Call"
        if self.tx_telegram == self.TELEGRAM_CALL_S_ROLL:
            return "Mode-S Roll-CAll"

        return "Mixed"

    """
    Taken from: https://github.com/junzis/pyModeS/blob/master/pyModeS/decoder/common.py
    """

    @staticmethod
    def hex2bin(hexstr,bitlength=None):
        """Convert a hexdecimal string to binary string, with zero fillings. """
        if not bitlength:
            binstr = bin(int(hexstr, 16))[2:]
        else:
            binstr =  bin(int(hexstr, 16))[2:].zfill(int(bitlength))
        return binstr

    @staticmethod
    def bin2int(binstr):
        """Convert a binary string to integer. """
        return int(binstr, 2)

    @staticmethod
    def hex2int(hexstr):
        """Convert a hexdecimal string to integer. """
        return int(hexstr, 16)

    @staticmethod
    def bin2np(binstr):
        """Convert a binary string to numpy array. """
        return np.array([int(i) for i in binstr])

    @staticmethod
    def np2bin(npbin):
        """Convert a binary numpy array to string. """
        return np.array2string(npbin, separator='')[1:-1]

    @staticmethod
    def df(msg):
        """Decode Downlink Format vaule, bits 1 to 5."""
        msgbin = hex2bin(msg)
        return min(bin2int(msgbin[0:5]), 24)

    @staticmethod
    def crc(msg, length, encode=False):
        """Mode-S Cyclic Redundancy Check
        Detect if bit error occurs in the Mode-S message
        Args:
            msg (string): 28 bytes hexadecimal message string
            encode (bool): True to encode the date only and return the checksum
        Returns:
            string: message checksum, or partity bits (encoder)
        """

        # the polynominal generattor code for CRC [1111111111111010000001001]
        generator = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
        ng = len(generator)

        #print("CRC: {}".format(msg))
        msgnpbin = sdr.bin2np(sdr.hex2bin(msg,length))
        #print(msg, msgnpbin,len(msgnpbin))
        if encode:
            msgnpbin[-24:] = [0] * 24
        #print(msgnpbin,len(msgnpbin))
        # loop all bits, except last 24 parity bits
        for i in range(len(msgnpbin) - 24):
            if msgnpbin[i] == 0:
                continue

            # perform XOR, when 1
            msgnpbin[i:i + ng] = np.bitwise_xor(msgnpbin[i:i + ng], generator)

        # last 24 bits
        reminder = sdr.np2bin(msgnpbin[-24:])
        #print(reminder)
        return reminder


'''


http://lists.ettus.com/pipermail/usrp-users_lists.ettus.com/2017-October/054579.html

    
from PyQt4 import QtCore,QtGui,uic
import sys
import libpyuhd as lib
import numpy as np
from multiprocessing import Process,Lock,Pool
import time


class MainWindows(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        uic.loadUi("wind.ui",self)
        self.connect(self.pushButton,QtCore.SIGNAL("clicked()"),self.start_on)
    def transmitter400(self):
        data = np.array(np.fft.fft(np.random.uniform(0, 1, 726)),
dtype=np.complex64)
        freq = 450e6
        print (freq)
        band = 50e6
        usrp = lib.usrp.multi_usrp("addr=192.168.10.2")
        st_args = lib.usrp.stream_args("fc32", "sc8")
        streamer = usrp.get_tx_stream(st_args)
        usrp.set_tx_rate(band, 0)
        usrp.set_tx_freq(lib.types.tune_request(freq), 0)
        usrp.set_tx_gain(40, 0)
        st_args.channels = [0]

        #data = np.array(np.random.uniform(0, 1, 4096), dtype=np.complex64)

        usrp.set_time_now(lib.types.time_spec(0))

        metadata = lib.types.tx_metadata()
        metadata.start_of_burst = True
        metadata.end_of_burst=False
        metadata.has_time_spec=True

        metadata.time_spec = lib.types.time_spec(0.1)
        metadata.time_spec+=usrp.get_time_now()

        # QtCore.QObject.connect(window.gener)
        while True:
            #QtCore.QThread.msleep(1)
            streamer.send(data, metadata)
            metadata.start_of_burst=False
            metadata.end_of_burst  = False
            metadata.has_time_spec = False
    def transmitter800(self):
        #lock = Lock()
        #lock.acquire()
        data1 = np.array(np.fft.fft(np.random.uniform(0, 1, 726)),
dtype=np.complex64)
        freq = 850e6
        print (freq)
        band = 50e6
        usrp1 = lib.usrp.multi_usrp("addr=192.168.10.3")
        st_args1 = lib.usrp.stream_args("fc32", "sc8")
        streamer1 = usrp1.get_tx_stream(st_args1)
        usrp1.set_tx_rate(band, 0)
        usrp1.set_tx_freq(lib.types.tune_request(freq), 0)
        usrp1.set_tx_gain(40, 0)
        st_args1.channels = [0]
        # data = np.array(np.random.uniform(0, 1, 4096), dtype=np.complex64)

        usrp1.set_time_now(lib.types.time_spec(0))

        metadata1 = lib.types.tx_metadata()
        metadata1.start_of_burst = True
        metadata1.end_of_burst = False
        metadata1.has_time_spec = True
        metadata1.time_spec = lib.types.time_spec(0.1)
        metadata1.time_spec += usrp1.get_time_now()

        # QtCore.QObject.connect(window.gener)
        while True:
#              QtCore.QThread.msleep(1)
              streamer1.send(data1, metadata1)
              metadata1.start_of_burst = False
              metadata1.end_of_burst = False
              metadata1.has_time_spec = False

            #print metadata
            #metadata.has_time_spec=False
            #if window.gener.isChecked() == True:
    def start_on(self):

            p = Process(target=self.transmitter400)
            p1 = Process(target=self.transmitter800)
            p.daemon=True
            p1.daemon=True
            p.start()
            p1.start()
            p.join()
            p1.join()
if  __name__=="__main__":
    app=QtGui.QApplication(sys.argv)
    window=MainWindows()
    window.show()
    sys.exit(app.exec_())
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
Tell me please, what seems to be a problem?


    
    '''
