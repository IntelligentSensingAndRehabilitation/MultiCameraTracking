# DHCP Server Setup

## Information needed for this setup

### Find your ethernet interface name

- Plug in an ethernet cable from the switch to the 10Gbit ethernet adapter on the computer
- In a terminal run:

```
 ip link show 
```

- The interface names will be listed, it will likely be something like enp#s0 where the # could be any number. This `interface_name` will be used for the remainder of the guide.
- If you know the MAC address for the ethernet adapter, you can match it to the interface name with that.
- If you don’t know the MAC and there are multiple enp names, try to unplug the cable going into the 10Gbit adapter and see which name disappears from the list

### Install DHCP Server package

```
sudo apt-get update
sudo apt-get install isc-dhcp-server
```

### Configure DHCP Server

Edit the DHCP configuration file to set up the DHCP service parameters:

```bash
sudo nano /etc/dhcp/dhcpd.conf
```

Set default and max lease time

```
default-lease-time 600;
max-lease-time 7200;
```

Configure subnet and IP range (`yourdomainname` can be set to anything, it is optional, I usually just put the computer host name) 

Add this block under the max-lease-time line. Be sure to update the `MAC Address for interface` in the host switch section.

```
subnet 192.168.1.0 netmask 255.255.255.0 {
    range 192.168.1.10 192.168.1.100;
    option domain-name-servers 8.8.8.8, 8.8.4.4;
    option domain-name "yourdomainname";
    option routers 192.168.1.1;
    option broadcast-address 192.168.1.255;
    
    host switch {
		    hardware ethernet MAC Address for interface;
		    fixed-address 192.168.1.2;
    }
}
```

### Configure network interface that will be used by DHCP

First open the file
```
sudo nano /etc/default/isc-dhcp-server
```
Then modify the following line with your `interface_name`
```
INTERFACESv4="interface_name"
```
### Configure Network Interface with nmcli

Replace `eth0` in the command below with interface_name below

```
nmcli con add type ethernet con-name DHCP-Server ifname eth0 autoconnect no ipv4.method manual ipv4.addresses 192.168.1.1/24 ipv4.gateway 192.168.1.1 ipv4.dns "8.8.8.8,8.8.4.4"
```

When you want to run the DHCP server:

```
nmcli con up DHCP-Server
sudo service isc-dhcp-server start
```