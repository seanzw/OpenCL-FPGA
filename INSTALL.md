# Install Guide
### Install CentOS 6.5
I already have a Windows 7, so I decided to install CentOS from hard disk.

- Boot into Windows, create a EXT3 partition to hold CentOS iso.
- Unpack `images` and `isolinux` from the iso to the root of EXT3 partition.
- Add a NeoGrub entry to boot from this partition.
- Reboot the machine, select NeoGrub entry to install CentOS.

Installation should go smooth without problem. Remember to choose Software Development Station when it asks you what this system is for, otherwise you won't have `gcc`, and it won't recognize NIC on my machine! So here is the problem: I need to download and install `gcc` to compile the NIC driver, but I won't have the internet until I install the driver. So, do select Software Development Station!

After installation, the first thing we have to do is get the NIC driver work. Mine is AR8161. You can search for yours to find a driver. To find out your network adapter's model, type `lspci | grep -i eth` in the terminal.

- Download and unpack the [alx driver](http://fichiers.touslesdrivers.com/34178/alx-linux-v2.0.0.6.rar).
- `make install`
- `cd src`
- `insmod alx.ko`

Now we have installed the driver. You can check this by `ifconfig -a` and you should see something like `eth0`. That's your network adapter!

However, by default, CentOS won't enable your NIC. You have to configure it by yourself. CentOS loads a configuration file for each NIC and we just need to modify it.

```
> vim etc/sysconfig/network-scripts/ifcfg-eth0

DEVICE=eth0
ONBOOT=yes
BOOTPROTO=dhcp
```

Notice that in my univerisity we use DHCP to allocate IP address. You can change it to static or other protocols.

### Install SDAccel 2015.3
Just follow the guide from Xilinx, there is no hiccup.

### Make lif easier!
Since I am going to use this system for a long time, it's time to make life more easier!

##### Add PATH to root.
I installed SDAccel with root user, which makes it a little tricky to add the binary folder into the PATH variable. After a litter searching, I found out that `/etc/skel/` is the skeleton for every user's home directory. Thus modifying `/etc/skel/.bashrc` means modifying every new created user's `.bashrc`, including root!

To add the path, open `/etc/skel/.bashrc` and add the following lines:
```
export PATH=/opt/Xilinx/SDAccel/2015.3/bin:$PATH
```
And now you can start `sdaccel` everywhere!
