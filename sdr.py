#!/usr/bin/env python

'''
 SDR-Class that abstacts the real SDR to the testdatagernerator
'''
import logging
from datetime import datetime, date
from time import time
from time import sleep

import uhd
import numpy as np
import uhd.libpyuhd as lib
import threading


class LogFormatter(logging.Formatter):
    """Log formatter which prints the timestamp with fractional seconds"""

    @staticmethod
    def pp_now():
        """Returns a formatted string containing the time of day"""
        now = datetime.now()
        return "{:%H:%M}:{:05.2f}".format(now, now.second + now.microsecond / 1e6)
        # return "{:%H:%M:%S}".format(now)

    def formatTime(self, record, datefmt=None):
        converter = self.converter(record.created)
        if datefmt:
            formatted_date = converter.strftime(datefmt)
        else:
            formatted_date = LogFormatter.pp_now()
        return formatted_date


class sdr:
    def __init__(self):
        self.usrp = None
        self.tx_level=None   #Current Level
        self.gain_offset = None
        self.tx_max_output_dbm = None # is set in set_channel


        self.threads = []
        self.quit_event = False
        self.sample_rate = 20e6
        self.frequency = -1
        self.sample_rate = -1
        self.tx_channel = -1
        self.tx_streamer = None
        self.tx_samples = None
        self.transmitting = False


        ''' Logger '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.console = logging.StreamHandler()
        self.logger.addHandler(self.console)
        formatter = LogFormatter(fmt='[%(asctime)s] [%(levelname)s] (%(threadName)-10s) %(message)s')
        self.console.setFormatter(formatter)

    def connect(self):
        self.logger.info("Connecting")
        self.usrp = uhd.usrp.MultiUSRP()
        self.usrp.set_time_now(lib.types.time_spec(0))
        self.logger.info("Using Device: %s", self.usrp.get_pp_string())

    def set_channel(self, frequency, tx_channel=0,sample_rate=10e6):
        if not self.usrp:
            self.connect()

        self.frequency = frequency
        self.tx_channel = tx_channel    # store for later use
        self.sample_rate = sample_rate  # store for later use

        self.usrp.set_tx_rate(self.sample_rate)
        #query the SDR to set the desired frequency
        self.usrp.set_tx_freq(lib.types.tune_request(self.frequency), )


        #setting the stream, numerical settings
        tx_cpu = "fc32"  # floatng complex 32bit
        tx_otw = "sc16"  # int complex 16bit
        st_args = uhd.usrp.StreamArgs(tx_cpu, tx_otw)
        st_args.channels = [self.tx_channel]

        #get the tx streamer for sending the data
        self.tx_streamer = self.usrp.get_tx_stream(st_args)

        '''
        #### Secondly Setup the RX channel aswell to receive the data loop:
        '''
        # Set the USRP rate, freq, and gain
        self.usrp.set_rx_rate(self.sample_rate, self.tx_channel)
        self.usrp.set_rx_freq(uhd.types.TuneRequest(self.frequency), self.tx_channel)




    def send_telegram(self, samples, level, future, repeat, delay):
        '''
        Sends out the created vector
        :param samples: The complex Vector to be send.
        :param level:   The RF Level
        :param future:  The inital delay
        :param repeat:  Number of times to repeat
        :param delay:   Delay between transmissions
        :return:
        '''

        '''
        The TX level needs to be calibated. This ist done using the SBP3 and an Splitter:
        The dB value for 0dB Gain is: -69.7341127997dBm
        '''
        # Todo: Raise Value errors for wrong levels:
        if level != self.tx_level:  # only on changes
            self.tx_level = level
            self.usrp.set_tx_gain(self.tx_level + 69.73411 + 1 ) #the headroom
            self.logger.info("Edit LVL {} {}".format(level, self.usrp.get_tx_gain()))


        leveled_tx_samples = samples *0.7943282 #Avoid Clipping of the sender, -1dBm
                                                            # #pow(10, (self.tx_level - self.tx_max_output_dbm) / 10.0) * self.tx_samples

        self.tx_buffer_len = len(samples) #used in receive_telegram

        metadata = uhd.types.TXMetadata()
        #This meatadata TimeSpec is used by receive_telegram to receive at the right moment
        if future != 0:
            metadata.has_time_spec = True
            self.current_time_spec = uhd.types.TimeSpec(self.usrp.get_time_now().get_real_secs() + future)
            metadata.time_spec = self.current_time_spec
        else:
            self.current_time_spec = False
            metadata.has_time_spec = False
        try:
            start_loop_time = datetime.utcnow()
            for i in range(repeat):
                start=datetime.utcnow()
                metadata.end_of_burst = True
                self.tx_streamer.send(leveled_tx_samples, metadata)
                if delay > 1/300:
                    """Receive and process the asynchronous TX messages"""
                    async_metadata = uhd.types.TXAsyncMetadata()
                    # Receive the async metadata, this returns the messages if the data is send successfully or not
                    while not self.tx_streamer.recv_async_msg(async_metadata, 0.1):
                        print("#",end='')
                    # Handle the error codes
                    if async_metadata.event_code == uhd.types.TXMetadataEventCode.burst_ack:
                        print(".", end='')
                    elif ((async_metadata.event_code == uhd.types.TXMetadataEventCode.underflow) or
                          (async_metadata.event_code == uhd.types.TXMetadataEventCode.underflow_in_packet)):
                        print("SEQ", end='')
                    elif ((async_metadata.event_code == uhd.types.TXMetadataEventCode.seq_error) or
                          (async_metadata.event_code == uhd.types.TXMetadataEventCode.seq_error_in_packet)):
                        print("UDR", end='')
                    elif (async_metadata.event_code == uhd.types.TXMetadataEventCode.time_error):
                        #if timeout occuered, reset time to start new.
                        #If it is more important to send the packets with a given rate, This error can be irgnored by commenting out this block
                        print("T", end='')
                        metadata.time_spec = uhd.types.TimeSpec(self.usrp.get_time_now().get_real_secs())
                    else:
                        self.logger.warning("Unexpected event on async recv (%s), continuing.",
                                       async_metadata.event_code)


                    #After the readback of the transmission the new TimeSpec is set
                    metadata.has_time_spec = True
                    metadata.time_spec = uhd.types.TimeSpec(metadata.time_spec.get_real_secs() + delay)
                else:
                    metadata.has_time_spec = False
                    sleep(delay)
        except RuntimeError as ex:
            self.logger.error("Runtime error in transmit: %s", ex)

        self.logger.info("\nTook: {}".format((datetime.utcnow() - start_loop_time).total_seconds()))

    def receive_telegram(self):
        usrp = self.usrp

        '''
        ### WARN: calibated for SBP3 Setup
        '''
        usrp.set_rx_gain(60, self.tx_channel)

        # Create the buffer to recv samples
        num_samps = int(self.tx_buffer_len * 1.15)
        # self.rx_samples = np.empty((1, num_samps), dtype=np.complex64)
        self.rx_samples = np.array([], dtype=np.complex64)

        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [self.tx_channel]

        metadata = uhd.types.RXMetadata()

        streamer = usrp.get_rx_stream(st_args)
        recv_buffer = np.zeros((1, num_samps), dtype=np.complex64)

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        #Set the TimeSpec to the time where the send_telegram set it
        if self.current_time_spec:
            stream_cmd.stream_now = False
            stream_cmd.time_spec = uhd.types.TimeSpec(self.current_time_spec.get_real_secs())
        else:
            stream_cmd.stream_now = True
        streamer.issue_stream_cmd(stream_cmd)
        # Receive the samples
        self.rx_samples = np.array([], dtype=np.complex64)

        recv_samps = 0
        while recv_samps < num_samps:

            samps = streamer.recv(recv_buffer, metadata)
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none and metadata.error_code != uhd.types.RXMetadataErrorCode.timeout:
                print(metadata.strerror())
                pass
            if samps:
                self.rx_samples = recv_buffer[0, :samps]
                recv_samps += samps
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stream_cmd)

        #Convert data into dBm
        np.seterr(divide = 'ignore')
        self.rx_samples_db = 10 * np.log10(10* (np.power(np.real(self.rx_samples),2)+np.power(np.imag(self.rx_samples),2))) - self.usrp.get_rx_gain()
        np.seterr(divide = 'warn')
