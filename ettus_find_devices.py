#!/usr/bin/env python

import uhd

print("Start searching")
usrp = uhd.usrp.MultiUSRP("type=b200")
usrp.set_rx_freq(100e6)

print(usrp)
