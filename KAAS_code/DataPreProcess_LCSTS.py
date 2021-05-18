#读取内容 进行分词
# 分词存储
import jieba
import jieba.analyse
import os
from fact_triple_extraction import _extraction_start
import sys
import re

# 文章、摘要 、最终生成文件的路径
ARTICL_PAHT = './dataset/article.txt'
SUMMARY_PATH = './dataset/summary.txt'
TRAIN_PAHT = './dataset/train.txt'
VAL_PAHT = './dataset/val.txt'
a = 2400000
# 读入文本并进行分词
def read_text_file(text_file):
    lines = []
    i = 1
    with open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            line = re.sub(r'\[.*\]', "", line)
            line = re.sub(r'\(.*\)', "", line)
            line = re.sub(r'#.*#', "", line)
            line = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', "", line)
            #line = re.sub(r'[a-z]*[:.]+\S+', "", line)
            line = re.findall('[\u4e00-\u9fa5a-zA-Z0-9，。《》“”：！？、]+', line, re.S)
            line = "".join(line)
            line = ' '.join(jieba.cut(line, HMM=False))
            lines.append(line.strip())
            i += 1
            if i%(a/10) == 0:
                print("%.2f%%" % (i*100/a))
            if i >= a:
                break
    return lines

# 对原文内容和摘要内容进行拼接
def mergeText(articlText,summaryText):
    trainList = []
    trainList_a = []
    valList = []
    valList_a = []
    linum = 0
    for seq1, seq2 in zip(articlText, summaryText):
        linum = linum + 1
        #print("%d" % linum)
        if linum%(a/50) == 0:
            print("%.2f%%" % (linum*100/a))
        seq1_aa = jieba.analyse.textrank(seq1, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
        seq1_a = []
        for i in seq1.split(' '):
            if i in seq1_aa and i not in seq1_a:
                seq1_a.append(i)
        seq1_a = ' '.join(seq1_a)
        seq1_a = seq1_a.strip()
        seq1_kg = _extraction_start(seq1.split(' '))
        #print(seq1_kg)
        if linum < a*1.8:
            seq1_a_len = len(seq1_a.split(' '))
            seq1_kg_len = len(seq1_kg.split(' '))
            seq2_len = len(seq2.split(' '))
            if seq1_a_len<=20 and seq1_a_len>=12 and seq1_kg_len>=10 and seq2_len >= 3:
                trainList.append(seq1)
                trainList.append(seq1_a)
                trainList.append(seq1_kg)
                trainList.append(seq2)
                #trainList_a.append(seq1_a)
            else:
                continue
        else:
            seq1_a_len = len(seq1_a.split(' '))
            seq1_kg_len = len(seq1_kg.split(' '))
            if seq1_a_len<=20 and seq1_a_len>=12 and seq1_kg_len>=10 and seq2_len >= 3:
                valList.append(seq1)
                valList.append(seq1_a)
                valList.append(seq1_kg)
                valList.append(seq2)
                #valList_a.append(seq1_a)
            else:
                continue


    return trainList,valList #,trainList_a, valList_a


#文本写入文件
def data_writer( finishList, path) :
    with open(path, 'w', encoding='utf-8') as writer:
        for item in finishList:
            writer.write(item+ '\n')


# 运行程序入口
if __name__ == '__main__':
    articls = read_text_file(ARTICL_PAHT)
    summays = read_text_file(SUMMARY_PATH)
    trainlist,vallist = mergeText(articls, summays)
    data_writer(trainlist,TRAIN_PAHT)
    data_writer(vallist,VAL_PAHT)
    #data_writer(trainList_a, TRAIN_A_PAHT)
    #data_writer(valList_a,VAL_A_PAHT)