sudo apt-get install libboost-all-dev libusb-1.0-0-dev python-mako doxygen python-docutils cmake build-essential cowsay lolcat
sudo apt install python3-pip
pip3 install numpy
git clone git://github.com/EttusResearch/uhd.git
cd uhd/host
mkdir build
cd build
cmake -DENABLE_PYTHON_API=ON -DENABLE_PYTHON3=ON -DENABLE_C_API=ON -DPYTHON_EXECUTABLE=/usr/bin/python3  -DCMAKE_INSTALL_PREFIX=/opt/uhd ../
make
make test
sudo make install
cd python
#Create venv:
python3 -m venv venv
#acivate it
source venv/bin/activate
#install uhd
sudo venv/bin/python3 setup.py install
pip install matplotlib

#Download the FPGA images:
sudo /usr/local/lib/uhd/utils/uhd_images_downloader.py

#install TKinter:
sudo apt-get install python3-tk


#Thread:
sudo groupadd usrp
sudo usermod -aG usrp $USER
#Then add the line below to end of the file /etc/security/limits.conf:
sudo echo "@usrp - rtprio  99" >> /etc/security/limits.conf


#Set udeev rules:
cd ../../utils
sudo cp uhd-usrp.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger

cowsay "Hey h_da crew! Installation of ettus_moc_up complete!" | lolcat --spread 1.0
