#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07 24 20:02:20 2019

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score


TrainStep = 10
TrainNum  = 100


# pickle 文件位置 / path of pickle files
pickle_path = './nlu/pickle'

# json文件位置 / path of json files
corpus_path = './nlu/corpus'


###############################################################
########### 语句文件预处理过程 / Pretreatment data process #######
###############################################################

def pre_data(data_file):

    # 定义空变量
    words     = []
    classes   = []
    documents = []
    responses = []
    slots     = []

    print(":) Pretreatment data file process:")
    
    # 导入语句文件 / Import statement file
    files= os.listdir(corpus_path)
    
    if files == None:
        print("Load json file failed!")
        return None
    
    print("Load json file success!")
    
    for file in files:
        with open(corpus_path + '/' + file) as json_data:
            statement = json.load(json_data)
            
            # 循环处理语句文件中的每个句子 / Loop through each statement in scenes
            for scene in statement['scenes']:
            
                # 收集tag信息到classes中 / Add tag to classes list
                if scene['tag'] not in classes:
                    classes.append(scene['tag'])
            
                # 收集queries和tag信息到documents中 / Add queries & tag to documents list
                for query in scene['queries']:
                    documents.append((query, scene['tag']))
            
                    # 将queries进行分词 / Word split in queries
                    query_words = jieba.cut(query)
                    # cut_text = smla_cut.cut(query, False)
                    # print(cut_text)
            
                    # 收集分词后每个词语到words中 / Add split word to word list
                    for query_word in query_words:
                        words.append(query_word)
            
                # 收集responses信息到responses中 / Add responses to responses list
                responses.append(scene['responses'])
            
                # 收集slot信息到slots中 / Add slot to slots list
                slots.append(scene['slot'])
    
    
    # 去除重复 / Remove duplicates
    words   = list(set(words))
    
    # 打印结果 / Print result
    
    print("Pretreatment Result:")
    print(len(words), "words:")
    print(words)
    print(len(classes), "classes:")
    print(classes)
    print(len(documents), "documents:")
    print(documents)
    print(len(responses), "responses:")
    print(responses)
    print(len(slots), "slots:")
    print(slots)
    print("")
    print("")
    
    # 返回数据list
    return words, classes, documents, responses, slots


##############################################################
############ 数字化过程 / Digitizing data process ##############
##############################################################
def dig_data(words, classes, documents, responses, slots):
    print(":) Digitizing data file process:")
    # 创建用于存储训练数据的缓存数组 / Create training data
    training = []
    
    # 数字化每个句子 / training set, data_x of words for each sentence
    for document in documents:
        
        pattern_word = []
        
        # 清空数组data_x / initialize data_x
        data_x = []
        
        # 提取document的句子 / list of tokenized words for the pattern
        pattern_words = document[0]
        # print(pattern_words)
        pattern_words_cut = jieba.cut(pattern_words)
        
        for query_word in pattern_words_cut:
            # print(query_word)
            pattern_word.append(query_word)
        # print(pattern_word)
        
        # 查看每个word是否在document中：是记为1；否记为0
        for word in words:
            if word in pattern_word:
                data_x.append(1)
            else:
                data_x.append(0)
        
        # 附加上每个document对应的class
        data_y = list([0] * len(classes))
        data_y[classes.index(document[1])] = 1
        
        # 将两者组合成一个数组
        training.append([data_x, data_y])
    
    # 打印出所有待训练的数据 / Print training data
    # print("training data: \n", training)
    
    # 将training的各行顺序打乱
    # random.shuffle(training)
    training = np.array(training)
    
    # train_x代表word在document中的对应关系；train_y代表document对应的class
    train_x = list(training[:,0])
    # print("training data x:")
    # for x in train_x: print(x)
    
    train_y = list(training[:,1])
    # print("training data y:")
    # for y in train_y: print(y)
    
    print("Data digitizing success!")
    print("")
    print("")
    
    # 返回两组数据
    return train_x, train_y

##############################################################
############ 训练模型过程 / Training modle process #############
##############################################################
def model_training(train_x, train_y):
    print(":) Training model process, please wait:")
    
    nums = np.arange(1,TrainNum,step=TrainStep)
    training_scores=[]
    for num in nums:
        clf=RandomForestClassifier(n_estimators=num)
        clf.fit(train_x,train_y)
        y_pred = clf.predict(train_x)
        training_scores.append(r2_score(train_y, y_pred))
    
    # ploat_result(training_scores)
    print(training_scores)
    
    max_num = np.argmax(training_scores)
    print("Max num of training:", max_num*TrainStep, "Score:", training_scores[max_num])
    
    # num = 100
    # clf = RandomForestClassifier(oob_score=False, max_depth=None, n_estimators=num) #(random_state=0, n_estimators=max_num, n_jobs=-1)
    # clf.fit(train_x, train_y)
        
    # y_pred = clf.predict(train_x)
    # print("")
    # print(":) Training score::", r2_score(train_y, y_pred))

    # print(clf.feature_importances_)
    return clf

##############################################################
############ 绘制训练结果 / Plot Training Result ###############
##############################################################
def ploat_result(training_scores):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(training_scores,label="Training Score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    # ax.set_xlim(0,100)
    ax.set_ylim(-1.1,1.1)
    plt.suptitle("RandomForestRegressor")
    # 设置 X 轴的网格线，风格为 点画线
    plt.grid(axis='x',linestyle='-.')
    plt.show()


##############################################################
############ 保存模型过程 / Save modle process #################
##############################################################
def save_model(clf):
    print("")
    print(":) Save model process:")
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)
    with open(pickle_path + '/model.pickle', 'wb') as f:
        pickle.dump(clf, f)
    print("Model saved!")
    return True


##############################################################
############ 保存语句处理结果过程 / Save statement process #######
##############################################################
def save_statement(words, classes, documents, responses, slots, train_x, train_y): 
    print("")
    print(":) Save statement process:")
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)
    with open(pickle_path + '/state.pickle', 'wb') as f:
        pickle.dump([words, classes, documents, responses, slots, train_x, train_y], f)
    print("Statement saved!")
    return True


##############################################################
###################### 主过程 / Main process ##################
##############################################################

print("HAI NLU -- A Natural Language Interaction Software.")
print("")


# 定义空变量 /
words     = []
classes   = []
documents = []

train_x   = []
train_y   = []

# 预处理 / 
(words, classes, documents, responses, slots) = pre_data(corpus_path)

# 数字化 / 
(train_x, train_y) = dig_data(words, classes, documents, responses, slots)

# 模型训练 /
clf = model_training(train_x, train_y)

# 保存模型 /
save_model(clf)

# 保存语句处理结果 /
save_statement(words, classes, documents, responses, slots, train_x, train_y)

print("")
print(":)Training done, exit!")
