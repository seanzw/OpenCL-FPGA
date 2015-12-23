# OpenCL-FPGA (Undergraduate Thesis)
Explores OpenCL on Xilinx's FPGA.

### Platform - CentOS 6.5 + SDAccel 2015.3
It took me quite some time to configure the platform, thus here I write down every steps I have taken.

##### Install CentOS 6.5
I already have a Windows 7, so I decided to install CentOS from hard disk.

- Boot into Windows, create a EXT3 partition to hold CentOS iso.
- Unpack `images` and `isolinux` from the iso to the root of EXT3 partition.
- Add a NeoGrub entry to boot from this partition.
- Reboot the machine, select NeoGrub entry to install CentOS.

Installation should go smooth without problem. Remember to choose Software Development Station when it asks you what this system is for, otherwise you won't have `gcc`, and it won't recognize the network adapter on my machine! So here is the problem: I need to download and install `gcc` to compile the network adapter driver, but I won't have the internet until I install the driver. So, do select Software Development Station!

After installation, the first thing we have to do is get the network adpater driver work. By default, CentOS won't enable the network adatper, so we have to configure it by ourselves.

My network adapter is XXX. You can search for yours to find a driver.

- Download and unpack the alx driver.
- `make install`
- `cd src`
- `insmod alx.ko`

Now we have installed the driver. You can check this by `ifconfig -a` and you should see something like `eth0`. That's your network adapter!

##### Install SDAccel 2015.3
Just follow the guide from Xilinx, there is no hiccup.

