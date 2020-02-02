#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07 25 21:25:34 2019

@project: HAI-NLU
@author : likw
@company: HuMan Inc.
"""

# 导入依赖库 / Imports
import jieba
import json
import pickle
import os
import numpy as np

from random import choice
#from sklearn.ensemble import RandomForestRegressor

version = '5.0.0'

model_file = './nlu/pickle/model.pickle'
state_file = './nlu/pickle/state.pickle'


##############################################################
######################       Initial      ####################
##############################################################
def initial():
    print("HAI NLU -- A Natural Language Interaction Software.")
    
    print(" _   _    __    ____ ")
    print("( )_( )  /__\  (_  _)")
    print(" ) _ (  /(__)\  _)(_ ")
    print("(_) (_)(__)(__)(____)")
    print("                     ")
    
    print("Version", version)
    print("Copyright © 2017-2020 HuMan Ltd.,Co.")
    print("All Rights Reserved.")
    print("https://www.smart-lands.com/hai-nlu")
    print("Type 'quit' to quit")
    print("")
    
    
##############################################################
########## 读取语句处理结果过程 / Load statement process ########
##############################################################
def load_statement():
    with open(state_file, 'rb') as f:
        (words, classes, documents, responses, slots, train_x, train_y) = pickle.load(f)
    print("Load state success!")
    return words, classes, documents, responses, slots, train_x, train_y


##############################################################
########### 读取模型过程 / Load model process ##################
##############################################################
def load_model():
    with open(model_file, 'rb') as f:
        clf = pickle.load(f)
    print("Load model success!")
    return clf


##############################################################
################## 关键词匹配 / Sentence match processing ######
##############################################################
def sentence_match(sentence, words, show_details=True):
    
    # 查看sentence中的词是否跟words中的关键词匹配 / generate probabilities
    # 对sentence进行分词 / tokenize the pattern
    sentence_words = jieba.cut(sentence)
    
    # 关键词匹配 / bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details: print ("   Found in bag   : %s" % w)
    
    # 查看匹配结果 / Match or not
    if 1 not in bag:
        match = False
        if show_details: print("   Predict result : No match!")
    else:
        match = True
        if show_details: print("   Predict result : Match")

    # 返回匹配结果 / return
    return(match, np.array(bag))


##############################################################
################# 响应过程 / answer process ###################
##############################################################
##############################################################
def answer(sentence, show_details=False):
    # 模型评估 /
    (match, match_arry) = sentence_match(sentence, words, show_details)
    
    match_predict = clf.predict([match_arry])
    p_array = match_predict[0].tolist()
    p_number = p_array.index(max(p_array))
    
    if show_details: print("   Probability    :", max(p_array) * 100, "%", classes[p_number])
    
    # if we have a classification then find the matching intent tag
    if match and max(p_array) > 0.6:
        # loop as long as there are matches to process
        slot_result   = slots[p_number]
        if show_details: print("   Slot select    :", slot_result)
        answer_result= choice(responses[p_number])
        return(answer_result, slot_result)
    else:
        answer_result = "听不懂"
        slot_result = ""
        return(answer_result, slot_result)


##############################################################
######################        Help        ####################
##############################################################
def help():
    for document in documents:
        print(document)


##############################################################
###################### 主过程 / Main process ##################
##############################################################
# 声明变量
words     = []
classes   = []
documents = []
responses = []
slots     = []

train_x   = []
train_y   = []

# 初始化打印
initial()

# 导入语句库
(words, classes, documents, responses, slots, train_x, train_y) = load_statement()

# 导入模型
clf = load_model()

# 自测试
print("----------------------------------------------")

answer_result,slot_result = answer("制造商", False)
print(":) " + answer_result)

print("----------------------------------------------")

