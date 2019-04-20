# ettus_moc_up
ETTUS B210 Moc up tool

## Install:

Windows

https://kb.ettus.com/Building_and_Installing_the_USRP_Open_Source_Toolchain_(UHD_and_GNU_Radio)_on_Windows


Linux

https://kb.ettus.com/Building_and_Installing_the_USRP_Open-Source_Toolchain_(UHD_and_GNU_Radio)_on_Linux


## Download:
Download this repo:
```
git clone https://github.com/slevon/ettus_moc_up.git
```



## Run:

change to the directory than active the virtaul environemnt
if on Linux rhe UHD is alread included in the environemnt

```
source venv/bin/activate
```

Verfiy if you can connect to SDR using Python:
```
python3 ettus_find_devices.py
```

Than run the testtool

```
python3 testdata_generator.py
```
