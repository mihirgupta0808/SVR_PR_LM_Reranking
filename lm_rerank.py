#!/usr/bin/env python
# coding: utf-8

# In[6]:


import operator
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import csv
import math
csv.field_size_limit(2147483647)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer() 
tokenizer = nltk.RegexpTokenizer(r"\w+")


# In[7]:



query_file_str = sys.argv[1]
#query_file_str = "queries.doctrain.tsv"
#top_100_file_str = "msmarco-doctrain-top100"
top_100_file_str = sys.argv[2]
collection_file_str = sys.argv[3]
#collection_file_str = "msmarco-docs.tsv"
model = sys.argv[4]


# In[12]:


collectionpickle = open(collection_file_str, "r",encoding="utf8")
docs = {}
docaddresses = {}
print("Collecting memory address")
doccount=0
curaddress = collectionpickle.tell()
while True:
    try:
        docalldata = collectionpickle.readline()
        docalldata = docalldata.split(sep="\t")
        docaddresses[docalldata[0]] = curaddress
        doccount+=1
        #if doccount == 50 :
            #break
        if  docalldata == None or docalldata[0] == None or len(docalldata[0]) <= 0:
            break    
        print("Doc Id : "+str(docalldata[0])+ " count is "+str(doccount))
        curaddress = collectionpickle.tell()
    except :
        break
    
        


# In[ ]:





# In[13]:


queries = {}
qcount = 0 
with open(query_file_str,encoding="utf8") as fd:
    rd = csv.reader(fd, delimiter="\t", quoting = csv.QUOTE_NONE)
    for row in rd:
        qno = row[0]
        qcount += 1
        #if qcount == 51:
            #break
        print("processing query :" + str(qcount) + ":" +  str(row[0]) )
        quer = tokenizer.tokenize(row[1])
        quer = [ps.stem(w.lower()) for w in quer if (not w.lower() in stop_words and w.lower().isalnum()) ]
        qworddict = {}
        for w in quer:
            if w not in qworddict:
                qworddict[w] = 1 
            else :
                qworddict[w] += 1
        queries[qno] = qworddict
        
    


# In[ ]:





# In[2]:


uniquetops = []
print("storing top docs")

qc = 0 
querytops = {}
with open(top_100_file_str,encoding="utf8") as ftop100:
    rtop100 = csv.reader(ftop100, delimiter = " ", quoting = csv.QUOTE_NONE)
    count = 0 
    for row in rtop100:
        count += 1
        if row[0] not in queries:
            continue
        if count % 100 == 1:
            #if qc == 50:
                #break
            qc += 1
            
            curqno = row[0]
            toparr = []
        toparr.append(row[2])
        if (row[2] not in uniquetops) and (curqno in queries):
            uniquetops.append(row[2])
        if count % 100 == 0:
            #if query in queries:
            querytops[curqno] = toparr
    


# In[1]:


fullvocab = []
docs = {}
totdocs = len(uniquetops)
print("processing docs ")
prc = 0 
for doc in uniquetops:
    print("processing doc " + str(prc) + "/" + str(totdocs) + " :  " + doc  )
    prc +=1 
    collectionpickle.seek(docaddresses[doc])
    docalldata = collectionpickle.readline() 
    docalldata = docalldata.split(sep="\t")
    new_title = tokenizer.tokenize(docalldata[2])
    new_title = [ps.stem(w.lower()) for w in new_title if (not w.lower() in stop_words and w.lower().isalnum())]
    new_content = tokenizer.tokenize(docalldata[3])
    new_content = [ps.stem(w.lower()) for w in new_content if not w in stop_words]
    worddict = {}
    for w in new_title:
        if w not in fullvocab:
            fullvocab.append(w)
        if w not in worddict:
            worddict[w] = 1
        else:
            worddict[w] += 1
    for w in new_content:
        if w not in fullvocab:
            fullvocab.append(w)
        if w not in worddict:
            worddict[w] = 1
        else:
            worddict[w] += 1
    arr = []
    arr.append(worddict)
    arr.append(len(new_title) + len(new_content))
    docs[docalldata[0]] = arr
    print("processed :" + str(prc) + " ,doc: " + docalldata[0] )


# In[ ]:


sms = {}
vocount = 0
vocsize = len(fullvocab)
ftcs = {}
Lc = 0.0
for word in fullvocab:
    vocount += 1
    print("smoothing : " + word + " : " + str(vocount) + "/"  + str(vocsize))
    fq = 0.0
    l = 0.0
    for doc in docs:
        l += docs[doc][1]
        if word in docs[doc][0]:
            fq += docs[doc][0][word]
    sms[word] = fq/l
    ftcs[word] = fq
    Lc = l


# In[ ]:


u = 1.5



# In[ ]:


if model == "uni":
    outfile =  open("unigramoutput.txt", "w")
    print("training uni")
    qcount = 0
    for query in querytops:
        qcount += 1
        print("training uni:" + query + " : " + str(qcount))
        docs100 = querytops[query]
        vocab = []
        for doc in docs100:
            for word in docs[doc][0]:
                if word not in vocab:
                    vocab.append(word)
        samqdict = queries[query]
        scores = {}
        pwrs = {}
        pwds = {}
        # pwds is dict of dict first word then doc
        for word in vocab:
            for doc in docs100:
                pwd = 0.0
                ftd = 0
                if word in docs[doc][0]:
                    ftd = docs[doc][0][word]
                D = docs[doc][1]
                #ftc = ftcs[word]
                pwd = (ftd + u*sms[word] )/(D + u)
                if word not in pwds:
                    wordoc = {}
                    wordoc[doc] = pwd
                    pwds[word] = wordoc
                else:
                    pwds[word][doc] = pwd
        pQds = {}


        for doc in docs100:
            D = docs[doc][1]
            pQd = 1.0
            for word in samqdict:
                pqd = 0.0
                fqd = 0
                if word in docs[doc][0]:
                    fqd = docs[doc][0][word]
                #ftc = ftcs[word]
                pqd = (fqd + u*sms[word] )/(D + u)
                if pqd != 0:
                    pQd *= pqd
                #pQd *= pqd
            pQds[doc] = pQd


        for word in vocab:
            pwr = 0.0
            for doc in docs100:
                pwr += pwds[word][doc]*pQds[doc]
            pwrs[word] = pwr


        scores = {}
        for doc in docs100:
            score = 0.0
            for word in vocab:
                score += pwrs[word]*pwds[word][doc]
            scores[doc] = score

        scores = dict( sorted(scores.items(), key=operator.itemgetter(1),reverse=True))
        curqno = query
        curank = 1
        for doc in scores:
            #print(str(curqno) + " Q0 " + str(doc) + " " + str(curank) + " " + str(rsvs[doc]) + " STANDARD\n" )
            outfile.write(str(curqno) + " Q0 " + str(doc) + " " + str(curank) + " " + str(scores[doc]) + " STANDARD\n"    )
            curank += 1


# In[ ]:


else:
    outfile =  open("bigramoutput.txt", "w")
    print("training bi")
    qcount = 0
    for query in querytops:
        qcount += 1
        print("training bi:" + query + " : " + str(qcount))
        docs100 = querytops[query]
        vocab = []
        for doc in docs100:
            for word in docs[doc][0]:
                if word not in vocab:
                    vocab.append(word)
        samqdict = queries[query]


        first = False
        prev = None
        pwdpairs = {}
        for word in vocab:
            if first == False:
                first = True
                for doc in docs100:
                    pwd = 0.0
                    ftd = 0
                    if word in docs[doc][0]:
                        ftd += docs[doc][0][word]
                    D = docs[doc][1]
                    ftc = ftcs[word]
                    pwd = (ftd + (u*(ftc)/Lc) )/(D + u)
                    if word not in pwdpairs:
                        wordoc = {}
                        wordoc[doc] = pwd
                        pwdpairs[word] = wordoc
                    else:
                        pwdpairs[word][doc] = pwd

            else:
                for doc in docs100:
                    lb = u/(u+docs[doc][1])
                    ftdcur = 0.0
                    if word in docs[doc][0]:
                        ftdcur += docs[doc][0][word]
                    D = docs[doc][1]
                    ftdprev = 0.0
                    if prev in docs[doc][0]:
                        ftdprev += docs[doc][0][prev]
                    if ftdprev == 0 or D == 0 or ftcs[prev] == 0 or Lc == 0 :
                        pwd = 0.0
                    else:
                        pwd = (1.0-lb)*( ((1.0-lb)*min(ftdcur,ftdprev))/ftdprev + (lb*ftdcur)/D  ) + lb*( ((1-lb)*min(ftcs[word],ftcs[prev]))/ftcs[prev] + (lb*ftcs[word])/Lc )
                    if word not in pwdpairs:
                        wordoc = {}
                        wordoc[doc] = pwd
                        pwdpairs[word] = wordoc
                    else:
                        pwdpairs[word][doc] = pwd


            prev = word


        pQdpairs = {}


        for doc in docs100:
            D = docs[doc][1]
            pQd = 1.0
            first = False
            prev = None
            for word in samqdict:

                if first == False:
                    first = True
                    pqd = 0.0
                    fqd = 0
                    if word in docs[doc][0]:
                        fqd = docs[doc][0][word]
                    ftc = ftcs[word]
                    pqd = (fqd + ((u*ftc)/Lc) )/(D + u)
                    if pqd != 0:
                        pQd *= pqd
                    #pQd *= pqd
                else:

                    lb = u/(u+docs[doc][1])
                    fqdcur = 0.0
                    if word in docs[doc][0]:
                        fqdcur += docs[doc][0][word]
                    D = docs[doc][1]
                    fqdprev = 0.0
                    if prev in docs[doc][0]:
                        fqdprev += docs[doc][0][prev]
                    if fqdprev == 0 or D == 0 or ftcs[prev] == 0 or Lc == 0 :
                        pqd = 0.0
                    else:
                        pqd = (1.0-lb)*( ((1.0-lb)*min(fqdcur,fqdprev))/fqdprev + (lb*fqdcur)/D  ) + lb*( ((1-lb)*min(ftcs[word],ftcs[prev]))/ftcs[prev] + (lb*ftcs[word])/Lc )
                    if pqd != 0:
                        pQd *= pqd


                prev = word
            pQdpairs[doc] = pQd
        pwrpairs = {}
        for word in vocab:
            pwr = 0.0
            for doc in docs100:
                pwr += pwdpairs[word][doc]*pQdpairs[doc]
            pwrpairs[word] = pwr

        pairscores = {}
        for doc in docs100:
            score = 0.0
            for word in vocab:
                score += pwrpairs[word]*pwdpairs[word][doc]
                #print(score)
            pairscores[doc] = score

        pairscores = dict( sorted(pairscores.items(), key=operator.itemgetter(1),reverse=True))
        curqno = query
        curank = 1
        for doc in pairscores:
            #print(str(curqno) + " Q0 " + str(doc) + " " + str(curank) + " " + str(pairscores[doc]) + " STANDARD\n" )
            outfile.write(str(curqno) + " Q0 " + str(doc) + " " + str(curank) + " " + str(pairscores[doc]) + " STANDARD\n"    )
            curank += 1


























    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




