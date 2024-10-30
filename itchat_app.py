# -*- coding:utf-8 -*-
import os
import re
import json
import time
from modules import gossip_robot,medical_robot,classifier
from utils.json_utils import dump_user_dialogue_context,load_user_dialogue_context

from wxauto import *
# 获取当前微信客户端


"""
问答流程：
1、用户输入文本
2、对文本进行解析得到语义结构信息
3、根据语义结构去查找知识，返回给用户

对文本进行解析的流程：
1、意图理解
    闲聊意图：问好、离开、肯定、拒绝
        问好、离开：需要有回复动作
        肯定、拒绝：需要执行动作
    诊断意图：
        当意图置信度达到一定阈值时(>=0.8)，可以查询该意图下的答案
        当意图置信度较低时(0.4~0.8)，按最高置信度的意图查找答案，询问用户是否问的这个问题
        当意图置信度更低时(<0.4)，拒绝回答
2、槽位填充
    如果输入是一个诊断意图，那么就需要语义槽的填充，得到结构化语义

"""



def delete_cache(file_name):
    """ 清除缓存数据，切换账号登入 """
    if os.path.exists(file_name):
        os.remove(file_name)


def text_replay(msg):
    # 判断是否是闲聊意图，以及是什么类型闲聊
    user_intent = classifier(msg[1])
    print(user_intent)
    if user_intent in ["greet","goodbye","deny","isbot"]:
        # 若为闲聊意图，则根据闲聊的类型，随机选择答案进行回答
        reply = gossip_robot(user_intent)
    elif user_intent == "accept":
        reply = load_user_dialogue_context(msg[0])
        reply = reply.get("choice_answer")
    else:#非闲聊意图，则调用意图识别服务
        reply = medical_robot(msg[1],msg[0])
        if reply["slot_values"]:
            dump_user_dialogue_context(msg[0],reply)
        reply = reply.get("replay_answer")
    return reply


if __name__ == '__main__':
    delete_cache(file_name='./logs/loginInfo.pkl')
    wx = WeChat()
    who = 'client'
    msg1 = wx.GetLastMessage
    reply=text_replay(msg1)
    wx.ChatWith(who)  # 打开`client`聊天窗口
    wx.SendMsg(reply)
    while True:#无限循环
        msg2 = wx.GetLastMessage
        if msg1 != msg2 and msg2[0]=="client":#当前后不同时，运行下面的命令
            reply=text_replay(msg2)
            wx.ChatWith(who) 
            wx.SendMsg(reply)
            msg1=msg2
