import os
import sys
import re
import spacy
import pytextrank
from openie import StanfordOpenIE
import networkx as nx
import numpy as np


A='./dataset_e_/'
ARTICL_PAHT = A+'article.txt'
SUMMARY_PATH = A+'summary.txt'
TRAIN_PAHT = A+'train.txt'
VAL_PAHT = A+'val.txt'
a = 2000
a2 = 0
aa = {}



G = nx.DiGraph()
client=StanfordOpenIE()

def read_text_file(text_file):
    lines = []
    i = 1
    with open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            #line = re.sub(r'\[.*\]', "", line)
            #line = re.sub(r'\(.*\)', "", line)
            #line = re.sub(r'#.*#', "", line)
            #line = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', "", line)
            line = line.lower()
            line = re.sub(r'[0-9]', "#", line)
            #line = " ".join(line)
            #line = ' '.join(jieba.cut(line, HMM=False))
            lines.append(line.strip())
            i += 1
            if i%(a/10) == 0:
                print("%.2f%%" % (i*100/a))
            if i >= a:
                break
    return lines
# 对原文内容和摘要内容进行拼接
def mergeText(articlText,summaryText):
    nlp = spacy.load("en_core_web_sm")
    tr = pytextrank.TextRank(logger=None)
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
    trainList = []
    trainList_a = []
    valList = []
    valList_a = []
    linum = 0
    for seq1, seq2 in zip(articlText, summaryText):
        linum = linum + 1
        if linum < a2:
            continue
        #print("%d" % linum)
        if linum%(a/50) == 0:
            print("%.2f%%" % (linum*100/a))
        aaa = nlp(seq1)
        seq1_aa = []
        for phrase in aaa.noun_chunks:
            seq1_aa.append(phrase.text)
        seq1_a = ' '.join(seq1_aa)
        #print(seq1_a)
        seq1_a = seq1_a.strip()
        seq1_kg = _extraction_start(seq1)
        #print(seq1_kg)
        if linum < -a*(2):
            seq1_a_len = len(seq1_a.split(' '))
            seq1_kg_len = len(''.join(seq1_kg.split(' ')))
            seq2_len = len(seq2.split(' '))
            if seq1_a_len>=12 and seq1_kg_len>=8 and seq2_len >= 3:
                trainList.append(seq1)
                trainList.append(seq1_a)
                trainList.append(seq1_kg)   
                trainList.append(seq2)
                #trainList_a.append(seq1_a)
            else:
                continue
        else:
            seq1_a_len = len(seq1_a.split(' '))
            seq1_kg_len = len(''.join(seq1_kg.split(' ')))
            seq2_len = len(seq2.split(' '))
            if seq1_a_len>=6 and seq1_kg_len>=8 and seq2_len >= 3:
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
        print(len(finishList))
        for item in finishList:
            writer.write(item+ '\n')


def _extraction_start(line):
    """
    事实三元组抽取的总控程序
    Args:
        in_file_name: 输入文件的名称
        #out_file_name: 输出文件的名称
        begin_line: 读文件的起始行
        end_line: 读文件的结束行
    """
    global aa, G
    sentence = line
    try:
        triples = fact_triple_extract(sentence)
    except:
        #print ("%d done" % (sentence_number))
        return ' '
    l = len(triples)
    i = 1
    while i < len(triples):
        ii = 0
        while ii < i:
            #print(i)
            #print(ii)
            if triples[i][0] == triples[ii][0] and triples[i][1] in triples[ii][1] and triples[i][1] != triples[ii][1]:
                #print(i)
                #print(ii)
                try:
                    triples.remove(triples[ii])
                    ii-=1
                    if i > 0:
                        i-=1
                except:
                    pass
                #ii-=1
            elif triples[i][0] == triples[ii][0] and triples[ii][1] in triples[i][1] and triples[i][1] != triples[ii][1]:
                try:
                    triples.remove(triples[i])
                    if i > 0:
                        i-=1
                except:
                    pass
                #i-=1
            elif triples[i][0]+triples[i][1]+triples[i][2] in triples[ii][0]+triples[ii][1]+triples[ii][2]:
                try:
                    triples.remove(triples[ii])
                    ii-=1
                    if i > 0:
                        i-=1
                except:
                    pass
                #ii-=1
            elif triples[ii][0]+triples[ii][1]+triples[ii][2] in triples[i][0]+triples[i][1]+triples[i][2]:
                try:
                    triples.remove(triples[i])
                    if i > 0:
                        i-=1
                except:
                    pass
                #i-=1
            elif triples[i][0] in triples[ii][0] and triples[i][1] in triples[ii][1] and triples[i][2] in triples[ii][2]:
                try:
                    triples.remove(triples[ii])
                    ii-=1
                    if i > 0:
                        i-=1
                except:
                    pass
                #ii-=1
            elif triples[ii][0] in triples[i][0] and triples[ii][1] in triples[i][1] and triples[ii][2] in triples[i][2]:
                try:
                    triples.remove(triples[i])
                    if i > 0:
                        i-=1
                except:
                    pass
                #i-=1
            ii += 1
        i+=1
    #print(triples2)
    edges = []  
    index = []
    for i in triples:
        index.append(10000*sentence.find(i[0])+100*sentence.find(i[1])+sentence.find(i[2]))
    index_2 = list(range(len(triples)))
    index = np.array([index_2, index])
    index = list(index.T[np.lexsort(index)].T)
    #print(index)
    for i in range(len(triples)):
        edges.append((triples[index[0][i]][0], triples[index[0][i]][1]))
        edges.append((triples[index[0][i]][1], triples[index[0][i]][2]))
    G.add_edges_from(edges)
    A = nx.dfs_tree(G)
    #nx.draw(G)
    G.clear()
    aa.clear()
    return ' '.join([str(i) for i in list(A)])
    





def fact_triple_extract(sentence):
    global client
    text = sentence
    #print('Text: %s.' % text)
    aaaa = []
    a = client.annotate(text)
    #print(a)
    for i in a:
        aaaa.append(list(i.values()))
    return aaaa
    
# 运行程序入口
if __name__ == '__main__':
    articls = read_text_file(ARTICL_PAHT)
    summays = read_text_file(SUMMARY_PATH)
    trainlist,vallist = mergeText(articls, summays)
    data_writer(trainlist,TRAIN_PAHT)
    data_writer(vallist,VAL_PAHT)
    #data_writer(trainList_a, TRAIN_A_PAHT)
    #data_writer(valList_a,VAL_A_PAHT)