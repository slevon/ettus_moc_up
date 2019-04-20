# ettus_moc_up
ETTUS B210 Moc up tool

## Install:

Windows

https://kb.ettus.com/Building_and_Installing_the_USRP_Open_Source_Toolchain_(UHD_and_GNU_Radio)_on_Windows


Linux

https://kb.ettus.com/Building_and_Installing_the_USRP_Open-Source_Toolchain_(UHD_and_GNU_Radio)_on_Linux


If the Python3 Lib is not created an alternative cmake command could be:
```
cmake -DENABLE_PYTHON_API=ON -DENABLE_PYTHON3=ON -DENABLE_C_API=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_LIBRARIES=/usr/lib/x86_64-linux-gnu/libpython3.6m.so ../
```

## Download:
Download this repo:
```
git clone https://github.com/slevon/ettus_moc_up.git
```



## Run:

change to the directory than active the virtaul environemnt (if on Linux)

```
source venv/bin/activate
```

than run the testtool

```
python3 testdata_generator.py
```
