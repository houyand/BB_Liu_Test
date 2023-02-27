#!/usr/bin/env python 
# -*- coding:utf-8 -*-
name = ['云', '悦', '众', '创', '微', '腾', '毅']
list = []
for x in name:
    for y in name:
        if x == y:
            continue
        for z in name:
            if y == z:
                continue
            list.append(x + y + z)

print(list)
