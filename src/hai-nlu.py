#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07 25 21:25:34 2019

@project: HAI-NLU
@author : likw
@company: HuMan Inc.
"""

# imports
import nlu.response as nlu
#import nlu.training as nlu
import act.action   as act


# debug
DEBUG = False #True #False


while True:
    
    # user input from terminal
    user_ask = input(":) ")

    # nlu
    if   user_ask == "quit" or user_ask == "quit()" or user_ask == "退出":
        print("Quit comand received!")
        quit()
    elif user_ask == "help" or user_ask == "help()" or user_ask == "帮助":
        nlu.help()
        slot_result = ''
    elif user_ask == "":
        continue
    else:
        answer_result,slot_result = nlu.answer(user_ask, DEBUG)
        print(":) " + answer_result)

    # act
    act_result= act.interface(slot_result)
    
