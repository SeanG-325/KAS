# coding=utf-8
"""
python *.py input.txt output.txt begin_line end_line
"""
# Set your own model path

MODELDIR="./ltp_data_v3.4.0/"
import sys

import os

import jieba

from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer

import re

import networkx as nx

segmentor = Segmentor()
segmentor.load(os.path.join(MODELDIR, "cws.model"))

postagger = Postagger()
postagger.load(os.path.join(MODELDIR, "pos.model"))

parser = Parser()
parser.load(os.path.join(MODELDIR, "parser.model"))

recognizer = NamedEntityRecognizer()
recognizer.load(os.path.join(MODELDIR, "ner.model"))


in_file_name = "input.txt"
out_file_name = "output.txt"
begin_line = 1
end_line = 0
aa = {}
G = nx.DiGraph()
if len(sys.argv) > 1:
    in_file_name = sys.argv[1]

if len(sys.argv) > 2:
    out_file_name = sys.argv[2]

if len(sys.argv) > 3:
    begin_line = int(sys.argv[3])

if len(sys.argv) > 4:
    end_line = int(sys.argv[4])

def extraction_start(in_file_name, out_file_name, begin_line, end_line):
    """
    事实三元组抽取的总控程序
    Args:
        in_file_name: 输入文件的名称
        #out_file_name: 输出文件的名称
        begin_line: 读文件的起始行
        end_line: 读文件的结束行
    """
    in_file = open(in_file_name, 'r', encoding='UTF-8')
    out_file = open(out_file_name, 'w')
    
    line_index = 1
    sentence_number = 0
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
        s = _extraction_start(sentence, out_file)
        out_file.write(s)
        #fact_triple_extract(sentence, out_file)
        out_file.flush()
        sentence_number += 1
        if sentence_number % 50 == 0:
            print ("%d done" % (sentence_number))
        text_line = in_file.readline()
        line_index += 1
    in_file.close()
    out_file.close()
    
def _extraction_start(line, out_file=None):
    """
    事实三元组抽取的总控程序
    Args:
        in_file_name: 输入文件的名称
        #out_file_name: 输出文件的名称
        begin_line: 读文件的起始行
        end_line: 读文件的结束行
    """
    sentence = line
    s = fact_triple_extract(sentence, out_file)
    #d = list(set(s.split(' ')))
    #words_d = {}
    #for i, a in enumerate(d):
    #    if a not in words_d.keys():
    #        words_d[a] = i
    a = 1
    edges = []
    for i in s.split(' '):
        if a % 2 == 1:
            a1 = i
            a += 1
        else:
            a2 = i
            #edges.append((words_d[a1], words_d[a2]))
            edges.append((aa[a1], aa[a2]))
            #edges.append((a1, a2))
            a += 1
    #print(edges)
    aa.clear()
    G.add_edges_from(edges)
    A = nx.dfs_tree(G)
    #nx.draw(G)
    G.clear()
    return ' '.join([str(i) for i in list(A)])
    


def fact_triple_extract(sentence, out_file=None):
    """
    对于给定的句子进行事实三元组抽取
    Args:
        sentence: 要处理的语句
    """
    s = ''
    #words = jieba.lcut(sentence)
    #words = sentence
    words = segmentor.segment(sentence)
    #print("\t".join(words))
    postags = postagger.postag(words)
    #print(postags)
    netags = recognizer.recognize(words, postags)
    arcs = parser.parse(words, postags)
    #print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    child_dict_list = build_parse_child_dict(words, postags, arcs)
    #print(len(postags))
    for index in range(len(postags)):
        # 抽取以谓词为中心的事实三元组
        if postags[index] == 'v':
            child_dict = child_dict_list[index]
            # 主谓宾
            if 'SBV' in child_dict.keys() and 'VOB' in child_dict.keys():
                e1_ = complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                r = words[index]
                e2_ = complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                e1 = words[child_dict['SBV'][0]]
                e2 = words[child_dict['VOB'][0]]
                if e1 not in aa.keys():
                    aa[e1] = ' '.join(jieba.lcut(e1_))
                if e2 not in aa.keys():
                    aa[e2] = ' '.join(jieba.lcut(e2_))
                if words[index] not in aa.keys():
                    aa[words[index]] = words[index]
                #out_file.write("(%s, %s, %s)\t" % (e1, r, e2)) #主语谓语宾语关系
                if out_file is None:
                    s += ("%s %s %s %s " % (e1, r, r, e2))
                else:
                    s += ("%s %s %s %s " % (e1, r, r, e2))
                    out_file.write("%s %s %s\n" % (e1, r, e2))
                    out_file.write("%s %s %s\n" % (e1_, r, e2_))
                    out_file.flush()
            # 定语后置，动宾关系
            if arcs[index].relation == 'ATT':
                if 'VOB' in child_dict.keys():
                    e1_ = complete_e(words, postags, child_dict_list, arcs[index].head - 1)
                    r = words[index]
                    e2_ = complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                    temp_string = r+e2_
                    if temp_string == e1_[:len(temp_string)]:
                        e1_ = e1_[len(temp_string):]
                    if temp_string not in e1_:
                        e1 = words[arcs[index].head - 1]
                        e2 = words[child_dict['VOB'][0]]
                        if e1 not in aa.keys():
                            aa[e1] = ' '.join(jieba.lcut(e1_))
                        if e2 not in aa.keys():
                            aa[e2] = ' '.join(jieba.lcut(e2_))
                        if words[index] not in aa.keys():
                            aa[words[index]] = words[index]
                        #out_file.write("(%s,%s,%s) " % (e1, r, e2)) #定语后置动宾关系
                        if out_file is None:
                            s += ("%s %s %s %s " % (e1, r, r, e2))
                        else:
                            s += ("%s %s %s %s " % (e1, r, r, e2))
                            out_file.write("%s %s %s\n" % (e1, r, e2))
                            out_file.write("%s %s %s\n" % (e1_, r, e2_))
                            out_file.flush()
            # 含有介宾关系的主谓动补关系
            if 'SBV' in child_dict.keys() and 'CMP' in child_dict.keys():
                #e1 = words[child_dict['SBV'][0]]
                e1_ = complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                cmp_index = child_dict['CMP'][0]
                r_ = words[index] + words[cmp_index]
                if 'POB' in child_dict_list[cmp_index].keys():
                    e2_ = complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                    e1 = words[child_dict['SBV'][0]]
                    r = words[index]
                    e2 = words[child_dict_list[cmp_index]['POB'][0]]
                    if e1 not in aa.keys():
                        aa[e1] = ' '.join(jieba.lcut(e1_))
                    if e2 not in aa.keys():
                        aa[e2] = ' '.join(jieba.lcut(e2_))
                    if words[index] not in aa.keys():
                        aa[words[index]] = ' '.join(jieba.lcut(r_))
                    #out_file.write("(%s,%s,%s) " % (e1, r, e2)) #介宾关系主谓动补
                    if out_file is None:
                        s += ("%s %s %s %s " % (e1, r, r, e2))
                    else:
                        s += ("%s %s %s %s " % (e1, r, r, e2))
                        out_file.write("%s %s %s\n" % (e1, r, e2))
                        out_file.write("%s %s %s\n" % (e1_, r, e2_))
                        out_file.flush()

        # 尝试抽取命名实体有关的三元组
        if netags[index][0] == 'S' or netags[index][0] == 'B':
            ni = index
            if netags[ni][0] == 'B':
                while netags[ni][0] != 'E':
                    ni += 1
                    if ni >= len(netags):
                        return ''
                e1_ = ''.join(words[index:ni+1])
            else:
                e1_ = words[ni]
            if arcs[ni].relation == 'ATT' and postags[arcs[ni].head-1] == 'n' and netags[arcs[ni].head-1] == 'O':
                r_ = complete_e(words, postags, child_dict_list, arcs[ni].head-1)
                if e1_ in r_:
                    r_ = r_[(r_.index(e1_)+len(e1_)):]
                if arcs[arcs[ni].head-1].relation == 'ATT' and netags[arcs[arcs[ni].head-1].head-1] != 'O':
                    e2_ = complete_e(words, postags, child_dict_list, arcs[arcs[ni].head-1].head-1)
                    mi = arcs[arcs[ni].head-1].head-1
                    li = mi
                    if netags[mi][0] == 'B':
                        while netags[mi][0] != 'E':
                            mi += 1
                            if mi >= len(netags):
                                return ''
                        e = ''.join(words[li+1:mi+1])
                        e2_ += e
                    if r_ in e2_:
                        e2_ = e2_[(e2_.index(r_)+len(r_)):]
                    if r_+ e2_ in sentence:
                        e1 = words[ni]
                        r = words[arcs[ni].head-1]
                        e2 = words[arcs[arcs[ni].head-1].head-1]
                        if e1 not in aa.keys():
                            aa[e1] = ' '.join(jieba.lcut(e1_))
                        if e2 not in aa.keys():
                            aa[e2] = ' '.join(jieba.lcut(e2_))
                        if words[arcs[ni].head-1] not in aa.keys():
                            aa[words[arcs[ni].head-1]] = ' '.join(jieba.lcut(r_))
                        #out_file.write("(%s,%s,%s) " % (e1, r, e2)) #人名//地名//机构
                        if out_file is None:
                            s += ("%s %s %s %s " % (e1, r, r, e2))
                        else:
                            s += ("%s %s %s %s " % (e1, r, r, e2))
                            out_file.write("%s %s %s\n" % (e1, r, e2))
                            out_file.write("%s %s %s\n" % (e1_, r, e2_))
                            out_file.flush()
    out_file.write("\n")
    return s
    
    out_file.write("\n")
    out_file.flush()

def build_parse_child_dict(words, postags, arcs):
    """
    为句子中的每个词语维护一个保存句法依存儿子节点的字典
    Args:
        words: 分词列表
        postags: 词性列表
        arcs: 句法依存列表
    """
    child_dict_list = []
    for index in range(len(words)):
        child_dict = dict()
        for arc_index in range(len(arcs)):
            if arcs[arc_index].head == index + 1:
                if arcs[arc_index].relation in child_dict.keys():
                    child_dict[arcs[arc_index].relation].append(arc_index)
                else:
                    child_dict[arcs[arc_index].relation] = []
                    child_dict[arcs[arc_index].relation].append(arc_index)
        #if child_dict.has_key('SBV'):
        #    print words[index],child_dict['SBV']
        child_dict_list.append(child_dict)
    return child_dict_list

def complete_e(words, postags, child_dict_list, word_index):
    """
    完善识别的部分实体
    """
    child_dict = child_dict_list[word_index]
    prefix = ''
    if 'ATT' in child_dict.keys():
        for i in range(len(child_dict['ATT'])):
           prefix += complete_e(words, postags, child_dict_list, child_dict['ATT'][i])
    
    postfix = ''
    if postags[word_index] == 'v':
        if 'VOB' in child_dict.keys():
            postfix += complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
        if 'SBV' in child_dict.keys():
            prefix = complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

    return prefix + words[word_index] + postfix

if __name__ == "__main__":
    #extraction_start(in_file_name, out_file_name, begin_line, end_line)
    extraction_start(in_file_name, out_file_name, begin_line, end_line)