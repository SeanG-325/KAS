# coding=utf-8
"""
python *.py input.txt output.txt begin_line end_line
"""
# Set your own model path
import numpy as np
from openie import StanfordOpenIE
import sys

import os

import re

import networkx as nx


in_file_name = "input.txt"
out_file_name = "output.txt"
begin_line = 1
end_line = 0
aa = {}
G = nx.DiGraph()
client=StanfordOpenIE()
if len(sys.argv) > 1:
    in_file_name = sys.argv[1]

if len(sys.argv) > 2:
    out_file_name = sys.argv[2]

if len(sys.argv) > 3:
    begin_line = int(sys.argv[3])

if len(sys.argv) > 4:
    end_line = int(sys.argv[4])
sentence_number = 0
def extraction_start(in_file_name, out_file_name, begin_line, end_line):
    global sentence_number
    in_file = open(in_file_name, 'r', encoding='UTF-8')
    out_file = open(out_file_name, 'w', encoding='utf-8')
    
    line_index = 1
    text_line = in_file.readline()
    while text_line:
        #print (text_line)
        if line_index < begin_line:
            text_line = in_file.readline()
            line_index += 1
            continue
        if end_line != 0 and line_index > end_line:
            break
        sentence = text_line.strip()
        if sentence == "" or len(sentence) > 1000:
            text_line = in_file.readline()
            line_index += 1
            continue
        s = _extraction_start(sentence)
        out_file.write(s+"\n")
        #fact_triple_extract(sentence, out_file)
        out_file.flush()
        sentence_number += 1
        if sentence_number % 50 == 0:
            print ("%d done" % (sentence_number))
        text_line = in_file.readline()
        line_index += 1
    in_file.close()
    out_file.close()
    
def _extraction_start(line, client=None):
    """
    事实三元组抽取的总控程序
    Args:
        in_file_name: 输入文件的名称
        #out_file_name: 输出文件的名称
        begin_line: 读文件的起始行
        end_line: 读文件的结束行
    """
    global aa
    sentence = line
    try:
        triples = fact_triple_extract(sentence, client)
    except:
        #print ("%d done" % (sentence_number))
        return ' '
    l = len(triples)
    i = 1
    while i < len(triples):
        ii = 0
        while ii < i:
            if triples[i][0] == triples[ii][0] and triples[i][1] in triples[ii][1] and triples[i][1] != triples[ii][1]:
                try:
                    triples.remove(triples[ii])
                    ii-=1
                except:
                    pass
                #ii-=1
            elif triples[i][0] == triples[ii][0] and triples[ii][1] in triples[i][1] and triples[i][1] != triples[ii][1]:
                try:
                    triples.remove(triples[i])
                    i-=1
                except:
                    pass
                #i-=1
            elif triples[i][0]+triples[i][1]+triples[i][2] in triples[ii][0]+triples[ii][1]+triples[ii][2]:
                try:
                    triples.remove(triples[ii])
                    ii-=1
                except:
                    pass
                #ii-=1
            elif triples[ii][0]+triples[ii][1]+triples[ii][2] in triples[i][0]+triples[i][1]+triples[i][2]:
                try:
                    triples.remove(triples[i])
                    i-=1
                except:
                    pass
                #i-=1
            elif triples[i][0] in triples[ii][0] and triples[i][1] in triples[ii][1] and triples[i][2] in triples[ii][2]:
                try:
                    triples.remove(triples[ii])
                    ii-=1
                except:
                    pass
                #ii-=1
            elif triples[ii][0] in triples[i][0] and triples[ii][1] in triples[i][1] and triples[ii][2] in triples[i][2]:
                try:
                    triples.remove(triples[i])
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
    





def fact_triple_extract(sentence, client_2=None):
    global client
    text = sentence
    #print('Text: %s.' % text)
    aaaa = []
    a = client.annotate(text)
    for i in a:
        aaaa.append(list(i.values()))
    return aaaa
    
    
    
    

if __name__ == "__main__":
    #extraction_start(in_file_name, out_file_name, begin_line, end_line)
    extraction_start(in_file_name, out_file_name, begin_line, end_line)
    