#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07 24 20:02:20 2018

@project: HAI-NLU
@author : likw
@company: HuMan Inc.
"""

# 导入依赖库 / Imports
# import webbrowser

def interface(slot_result):
    if slot_result == "HAI-LIGHT-A":
        print("需要添加开灯指令。")
        return True
    elif slot_result == "HAI-LIGHT-B":
        print("需要添加关灯指令。")
        return True
    else:
        # print("这个动作无法执行。")
        return False
