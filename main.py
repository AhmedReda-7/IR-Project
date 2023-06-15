import pandas as pd
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import operator


def getInputFiles(filelist):  # the funcation that read all paths in the file
    with open(filelist) as f:
        return [a for a in f.read().split()]


def preprocess(data):  # the funcation that remove punctuation
    for p in "!.,:@#$%^&?<>*()[}{]-=;/\"\\\t\n":
        if p in '\n;?:!.,.':
            data = data.replace(p, ' ')
        else:
            data = data.replace(p, '')
    return data.lower()


def computeIDF(docList):  # the funcation that compute idf
    idf = {}
    N = 10  # the number of doc
    for word, val in docList.items():
        if N == val:  # check if num of doc =number of tf
            idf[word] = 1
        else:
            idf[word] = math.log10(N / float(val))

    return idf


def computeTF(wordDict):  # compute the tf weight
    tfDict = {}
    for word, count in wordDict.items():
        if count > 0:
            tfDict[word] = 1 + math.log10(float(count))
        else:
            tfDict[word] = 0
    return tfDict


def computeTFIDF(tfw, idfs):
    tfidf = {}
    for word, val in tfw.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def compare(r, s):
    if len(s) >= len(r):
        for y in range(len(r)):
            for t in range(len(s)):
                if r[y] + 1 == s[t]:
                    return True
                else:
                    continue

    else:
        for y in range(len(s)):
            for t in range(len(r)):
                if r[t] + 1 == s[y]:
                    return True
                else:
                    continue


print('------------------------------------------------------------------------------------------------------------')
files = getInputFiles("input.txt")  # get list of all paths
filenumber = 1  # intialize the file number=1
index = {}  # the dectionary than had all item and his postion
doc_dic = {}  # this dic will contain the number of file and the bath
for i in range(len(files)):
    doc_dic[i + 1] = files[i]
# list of stop words
stop_words = ['was', 'what', "haven't", "weren't", 'above', 'yours', 'aren', 'herself', 'y', 'me', "she's", 'the',
              "aren't", 'having', 'not', 'her', 'than', 'how', 's', 'o', 'should', 'at', 'when', 'have', 'of', 'under',
              'themselves', 'will', 'had', "you'd", 'ours', 'he', 'while', 'with', "doesn't", 'its', 'm', 'are',
              "needn't", 'we', 'haven', 'as', 'wouldn', 'ourselves', 'hers', "you've", 'some', 'from', 'couldn',
              'yourself', 'hadn', 'again', 'you', 'below', 'those', 'who', 'this', 'a', 'his', 'nor', 'don', 'needn',
              'no', 'if', 'then', 'so', 'off', 't', 'any', "isn't", 'just', 'him', "mustn't", 'against', 'only', 'our',
              'i', "it's", 'very', 'isn', "shan't", 'until', 'is', 'an', 'why', 'other', 'same', 'now', 'few', "hasn't",
              'down', 'over', 'd', 'by', "shouldn't", 're', "mightn't", "don't", 'such', 'out', 'shouldn', 'about',
              'there', "you'll", 'hasn', "won't", 'mightn', 'mustn', 'did', 'didn', 'these', 'were', "should've",
              'most', 'wasn', 'be', 'do', 'all', 'their', "that'll", 'through', 'which', 'myself', 'further', 'on',
              'theirs', 'weren', 'during', 'for', 'that', 'does', "couldn't", "hadn't", 'am', 'here', 'up', 'shan',
              'won', 'because', 'has', "you're", 'doing', "didn't", 'doesn', 'own', 'each', 'them', 'more', 'both',
              'she', "wasn't", 'between', 'itself', 'too', 'll', 'himself', 'into', 'being', 'can', 'my', 'once', 'ain',
              'your', 'or', "wouldn't", 'ma', 'it', 'been', 've', 'after', 'before', 'whom', 'yourselves', 'but',
              'they', 'and']
for i in range(len(files)):
    with open(files[i]) as f:
        l = f.read()  # read all things in the doc
        doc = [a for a in preprocess(l).split()]  # remove the punc and split the doc and put the token in list
        f1 = [w for w in doc if not w.lower() in stop_words]  # change all item to lower and remove the stopwords
        for idx, word in enumerate(f1):  # use to get the postion of the all items in list
            if word in index:  # check if the token is in the index
                if filenumber in index[word][0]:  # check if the file number in the index
                    index[word][0][filenumber].append(idx + 1)  # insert the postion in the file number of token

                else:
                    index[word][0][filenumber] = [idx + 1]  # insert the file numer and the first postion

            else:
                index[word] = []  # initialize the list.
                index[word].append({})  # initialize the dic that contain The postings list is initially empty.
                index[word][0][filenumber] = [idx + 1]  # Add doc number and postion .
        filenumber += 1  # Increment the file number
print("the token is")
for l in index.keys():  # this for loop will print all token
    print(l)
print("------------------------------------------------------------------------------------------------------------")
diction = {}  # the dic that contain every token and the number doc that contain the token
for k, l in sorted(index.items()):  # using to print every token and his postion
    s = len(l[0])  # the number of doc
    print(k, ":", s, ":", l)
    diction[k] = int(s)  # add token and the number doc that contain the token
print("------------------------------------------------------------------------------------------------------------")
# print("the df of all token", diction)
idf = computeIDF(diction)
# print("the idf is",idf)
df1 = pd.DataFrame.from_dict([diction, idf]).T
df1.columns = ["DF", "IDF"]
print(df1, "\n ---------------------------------------------------------")
dic1 = {}  # compute the tf
for l, k in sorted(index.items()):
    dic1[l] = 0
    for j, n in k[0].items():
        if j == 1:
            dic1[l] = len(n)
dic2 = {}
for l, k in sorted(index.items()):
    dic2[l] = 0
    for j, n in k[0].items():
        if j == 2:
            dic2[l] = len(n)
dic3 = {}
for l, k in sorted(index.items()):
    dic3[l] = 0
    for j, n in k[0].items():
        if j == 3:
            dic3[l] = len(n)
dic4 = {}
for l, k in sorted(index.items()):
    dic4[l] = 0
    for j, n in k[0].items():
        if j == 4:
            dic4[l] = len(n)
dic5 = {}
for l, k in sorted(index.items()):
    dic5[l] = 0
    for j, n in k[0].items():
        if j == 5:
            dic5[l] = len(n)
dic6 = {}
for l, k in sorted(index.items()):
    dic6[l] = 0
    for j, n in k[0].items():
        if j == 6:
            dic6[l] = len(n)
dic7 = {}
for l, k in sorted(index.items()):
    dic7[l] = 0
    for j, n in k[0].items():
        if j == 7:
            dic7[l] = len(n)
dic8 = {}
for l, k in sorted(index.items()):
    dic8[l] = 0
    for j, n in k[0].items():
        if j == 8:
            dic8[l] = len(n)
dic9 = {}
for l, k in sorted(index.items()):
    dic9[l] = 0
    for j, n in k[0].items():
        if j == 9:
            dic9[l] = len(n)
dic10 = {}
for l, k in sorted(index.items()):
    dic10[l] = 0
    for j, n in k[0].items():
        if j == 10:
            dic10[l] = len(n)
print(
    "\n                                     the term Frequency for each term in each document                        \n")
df2 = pd.DataFrame.from_dict([dic1, dic2, dic3, dic4, dic5, dic6, dic7, dic8, dic9, dic10]).T
df2.columns = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]
print(df2, "\n ---------------------------------------------")
dictfw1 = computeTF(dic1)  # compute the tf weight of each doc
dictfw2 = computeTF(dic2)
dictfw3 = computeTF(dic3)
dictfw4 = computeTF(dic4)
dictfw5 = computeTF(dic5)
dictfw6 = computeTF(dic6)
dictfw7 = computeTF(dic7)
dictfw8 = computeTF(dic8)
dictfw9 = computeTF(dic9)
dictfw10 = computeTF(dic10)
print(
    "\n                                     the term Frequency weight for each term in each document                        \n")
df3 = pd.DataFrame.from_dict(
    [dictfw1, dictfw2, dictfw3, dictfw4, dictfw5, dictfw6, dictfw7, dictfw8, dictfw9, dictfw10]).T
df3.columns = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]
print(df3, "\n -----------------------------------------------------")
dictf_idf1 = computeTFIDF(dictfw1, idf)
dictf_idf2 = computeTFIDF(dictfw2, idf)
dictf_idf3 = computeTFIDF(dictfw3, idf)
dictf_idf4 = computeTFIDF(dictfw4, idf)
dictf_idf5 = computeTFIDF(dictfw5, idf)
dictf_idf6 = computeTFIDF(dictfw6, idf)
dictf_idf7 = computeTFIDF(dictfw7, idf)
dictf_idf8 = computeTFIDF(dictfw8, idf)
dictf_idf9 = computeTFIDF(dictfw9, idf)
dictf_idf10 = computeTFIDF(dictfw10, idf)
print("\n                                     the tf.idf for each term in each document                        \n")
df4 = pd.DataFrame.from_dict(
    [dictf_idf1, dictf_idf2, dictf_idf3, dictf_idf4, dictf_idf5, dictf_idf6, dictf_idf7, dictf_idf8, dictf_idf9,
     dictf_idf10]).T
df4.columns = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]
print(df4, "\n ---------------------------------------------------------")
pow_len1 = 0
for k, w in dictf_idf1.items():  # compute the lenth of each doc
    pow_len1 += pow(w, 2)
d1_len = math.sqrt(pow_len1)
pow_len2 = 0
for k, w in dictf_idf2.items():
    pow_len2 += pow(w, 2)
d2_len = math.sqrt(pow_len2)
pow_len3 = 0
for k, w in dictf_idf3.items():
    pow_len3 += pow(w, 2)
d3_len = math.sqrt(pow_len3)
pow_len4 = 0
for k, w in dictf_idf4.items():
    pow_len4 += pow(w, 2)
d4_len = math.sqrt(pow_len4)
pow_len5 = 0
for k, w in dictf_idf5.items():
    pow_len5 += pow(w, 2)
d5_len = math.sqrt(pow_len5)
pow_len6 = 0
for k, w in dictf_idf6.items():
    pow_len6 += pow(w, 2)
d6_len = math.sqrt(pow_len6)
pow_len7 = 0
for k, w in dictf_idf7.items():
    pow_len7 += pow(w, 2)
d7_len = math.sqrt(pow_len7)
pow_len8 = 0
for k, w in dictf_idf8.items():
    pow_len8 += pow(w, 2)
d8_len = math.sqrt(pow_len8)
pow_len9 = 0
for k, w in dictf_idf9.items():
    pow_len9 += pow(w, 2)
d9_len = math.sqrt(pow_len9)
pow_len10 = 0
for k, w in dictf_idf10.items():
    pow_len10 += pow(w, 2)
d10_len = math.sqrt(pow_len10)
print("\n                                     the doc len for each term in each document                        \n")
df5 = pd.DataFrame.from_dict([d1_len, d2_len, d3_len, d4_len, d5_len, d6_len, d7_len, d8_len, d9_len, d10_len]).T
df5.columns = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]
print(df5, "\n --------------------------------------")
nor_dic1 = {}
for l, k in dictf_idf1.items():
    nor_dic1[l] = k / d1_len
nor_dic2 = {}
for l, k in dictf_idf2.items():
    nor_dic2[l] = k / d2_len
nor_dic3 = {}
for l, k in dictf_idf3.items():
    nor_dic3[l] = k / d3_len
nor_dic4 = {}
for l, k in dictf_idf4.items():
    nor_dic4[l] = k / d4_len
nor_dic5 = {}
for l, k in dictf_idf5.items():
    nor_dic5[l] = k / d5_len
nor_dic6 = {}
for l, k in dictf_idf6.items():
    nor_dic6[l] = k / d6_len
nor_dic7 = {}
for l, k in dictf_idf7.items():
    nor_dic7[l] = k / d7_len
nor_dic8 = {}
for l, k in dictf_idf8.items():
    nor_dic8[l] = k / d8_len
nor_dic9 = {}
for l, k in dictf_idf9.items():
    nor_dic9[l] = k / d9_len
nor_dic10 = {}
for l, k in dictf_idf10.items():
    nor_dic10[l] = k / d10_len
print(
    "\n                                     the normalized tf.idf for each term in each document                        \n")
df6 = pd.DataFrame.from_dict(
    [nor_dic1, nor_dic2, nor_dic3, nor_dic4, nor_dic5, nor_dic6, nor_dic7, nor_dic8, nor_dic9, nor_dic10]).T
df6.columns = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]
print(df6, "\n -----------------------------------------------------")
x = input("enter query phase:")
query = []
f1 = [a for a in preprocess(x).split()]  # remove the punc and split the doc and put the token in list
oop = [w for w in f1 if not w.lower() in stop_words]
for j in oop:  # check if the elements of the query in the token
    for l in index.keys():
        if j == l:
            query.append(j)
if len(oop) > len(query):
    print("no matched files")
else:
    h = []  # choose the doc of each item in query
    for l, k in sorted(index.items()):
        for i in range(len(sorted(query))):
            if l == query[i]:
                o = []
                for j, n in k[0].items():
                    o.append(j)

                h.append(o)
    # print(h)
    p = sorted(list(set.intersection(*map(set, h))))  # choose the intesection doc
    # print(p)
    # give us all postion of the match files
    hpp = []
    for u in p:
        for c in range(len(query)):
            for l, k in sorted(index.items()):
                for j, n in k[0].items():
                    if l == query[c]:
                        if j == u:
                            hpp.append(n)
                            # print(hpp)
    # check the len of query to get the file matched
    if len(query) == 1:
        print("the matched files is ")
        for k in p:
            print(files[k - 1])
        query_tf = {}  # we will put the query frequency
        for l, k in sorted(index.items()):  # using to put all token =0
            query_tf[l] = 0
        for o in query:  # using to make for loop in the query
            for l, k in sorted(query_tf.items()):  # using for loop in the query_tf
                if o == l:  # check if the item in the query=item in query_tf
                    query_tf[l] = k + 1  # add 1 to the value in the query_tf
                else:
                    query_tf[l] = k
        print("---------------------------------------------------------------------------------------")
        print("the tf of query", query_tf)
        print("---------------------------------------------------------------------------------------")
        query_tfw = computeTF(query_tf)  # calaulate the query frequency weight
        print("the tfw of query", query_tfw)
        print("------------------------------------------------------------------------------------")
        tf_idf_query = computeTFIDF(query_tfw, idf)  # calaulate the tf.idf query
        print("the tf.idf of query", tf_idf_query)
        print("----------------------------------------------------------------------------------------")
        pow_len_query = 0
        for k, w in tf_idf_query.items():
            pow_len_query += pow(w, 2)
        query_len = math.sqrt(pow_len_query)
        print("the query len is:", query_len)
        print("-----------------------------------------------------------------------------------------------")
        nor_query = {}
        for l, k in tf_idf_query.items():
            nor_query[l] = k / query_len
        print("the normalization of query is:", nor_query)
        print("------------------------------------------------------------------------------------------------")
        dic_of_match = {}  # get the doc that match with phase query
        for i in p:
            for u, k in doc_dic.items():
                if i == u:
                    dic_of_match[i] = k
                else:
                    continue

        his = [nor_dic1, nor_dic2, nor_dic3, nor_dic4, nor_dic5, nor_dic6, nor_dic7, nor_dic8, nor_dic9,
               nor_dic10]  # put all tf idf in list
        hjh = {}  # put the number of the file to the tf.idf dic
        for u, k in doc_dic.items():
            hjh[u] = his[u - 1]
        lll = {}  # match the doc that match with phase query with tf.idf
        for i in p:
            for u, k in hjh.items():
                if i == u:
                    lll[i] = k
                else:
                    continue
        jjhh = []  # put the value of tf.idf that match with the phase query
        for u, k in lll.items():
            jjhh.append(list(k.values()))
        alle = list(nor_query.values())  # put the value of tf.idf normalize of query
        tf_idf_query_val = np.array(alle)
        s = []  # list of all similarity
        for i in jjhh:
            a = np.array(i)
            result = cosine_similarity(a.reshape(1, -1), tf_idf_query_val.reshape(1, -1))
            s.append(result)
        ss = {}  # the dic that put number of file with his similarity
        for i in range(len(s)):
            ss[p[i]] = s[i]
        print("each doc with his similarity", ss)
        # sort the similarity
        sorted_d = dict(sorted(ss.items(), key=operator.itemgetter(1), reverse=True))
        print("the doc after sorted", sorted_d)
        php = {}  # the dic that has Rank documents based on cosine similarity
        for u, k in sorted_d.items():
            for p, l in doc_dic.items():
                if u == p:
                    php[u] = l
        print("the Rank documents based on cosine similarity ")
        for i, f in php.items():
            print(f)
    else:
        h = int(0)
        gggj = []
        for y in range(int(len(p))):
            for t in range(int(len(query) - 1)):
                if compare(hpp[t + h], hpp[t + 1 + h]) == True:
                    gggj.append(True)
                else:
                    gggj.append(False)
            if len(p) == 1:
                continue
            else:
                h += int(len(query))
        tt = []  # the doc that his postion match
        cc = []  # the doc that his postion no match
        # print(gggj)
        uul = int(0)
        for u in range(len(p)):
            for i in range(len(query) - 1):
                if gggj[i + uul] == True:
                    continue
                else:
                    cc.append(p[u])

            uul += int(len(query) - 1)
        if len(cc) == 0:
            tt = p
        else:
            tt = list(set(p) - set(cc))
        print("the doc", tt)
        print("---------------------------")
        if len(tt) == 0:
            print("no matched files")
        else:
            print("the matched files is:")
            for k in tt:
                print(files[k - 1])

            query_tf = {}  # we will put the query frequency
            for l, k in sorted(index.items()):  # using to put all token =0
                query_tf[l] = 0
            for o in query:  # using to make for loop in the query
                for l, k in sorted(query_tf.items()):  # using for loop in the query_tf
                    if o == l:  # check if the item in the query=item in query_tf
                        query_tf[l] = k + 1  # add 1 to the value in the query_tf
                    else:
                        query_tf[l] = k
            print("the tf of query", query_tf)
            print("---------------------------------------------------------------------------------------")
            query_tfw = computeTF(query_tf)  # calaulate the query frequency weight
            print("the tfw of query", query_tfw)
            print("------------------------------------------------------------------------------------")
            tf_idf_query = computeTFIDF(query_tfw, idf)  # calaulate the tf.idf query
            print("the tf.idf of query", tf_idf_query)
            print("----------------------------------------------------------------------------------------")
            pow_len_query = 0
            for k, w in tf_idf_query.items():
                pow_len_query += pow(w, 2)
            query_len = math.sqrt(pow_len_query)
            print("the query len is:", query_len)
            print("-----------------------------------------------------------------------------------------------")
            nor_query = {}
            for l, k in tf_idf_query.items():
                nor_query[l] = k / query_len
            print("the normalization of query is:", nor_query)
            print("------------------------------------------------------------------------------------------------")
            dic_of_match = {}  # get the doc that match with phase query
            for i in tt:
                for u, k in doc_dic.items():
                    if i == u:
                        dic_of_match[i] = k
                    else:
                        continue

            his = [nor_dic1, nor_dic2, nor_dic3, nor_dic4, nor_dic5, nor_dic6, nor_dic7, nor_dic8, nor_dic9,
                   nor_dic10]  # put all tf idf in list
            hjh = {}  # put the number of the file to the tf.idf dic
            for u, k in doc_dic.items():
                hjh[u] = his[u - 1]
            lll = {}  # match the doc that match with phase query with tf.idf
            for i in tt:
                for u, k in hjh.items():
                    if i == u:
                        lll[i] = k
                    else:
                        continue
            jjhh = []  # put the value of tf.idf that match with the phase query
            for u, k in lll.items():
                jjhh.append(list(k.values()))
            alle = list(nor_query.values())  # put the value of tf.idf normalize of query
            tf_idf_query_val = np.array(alle)
            s = []  # list of all similarity
            for i in jjhh:
                a = np.array(i)
                result = cosine_similarity(a.reshape(1, -1), tf_idf_query_val.reshape(1, -1))
                s.append(result)
            ss = {}  # the dic that put number of file with his similarity
            for i in range(len(s)):
                ss[tt[i]] = s[i]
            print("each doc with his similarity", ss)
            # sort the similarity
            sorted_d = dict(sorted(ss.items(), key=operator.itemgetter(1), reverse=True))
            print("the doc after sorted", sorted_d)
            php = {}  # the dic that has Rank documents based on cosine similarity
            for u, k in sorted_d.items():
                for p, l in doc_dic.items():
                    if u == p:
                        php[u] = l
            print("the Rank documents based on cosine similarity ")
            for i, f in php.items():
                print(f)