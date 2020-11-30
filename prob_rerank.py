#!/usr/bin/env python
# coding: utf-8

# In[1]:


import operator
import sys
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math 
import pickle
csv.field_size_limit(2147483647)
expansion_limit = 10
query_file_str = sys.argv[1]
top_100_file_str = sys.argv[2]
collection_file_str = sys.argv[3]
expansion_limit = int(sys.argv[4])


stop_words = set(stopwords.words('english'))
ps = PorterStemmer() 
print(stop_words)
tokenizer = nltk.RegexpTokenizer(r"\w+")


# In[ ]:


#collection_file_str = "msmarco-docs.tsv"
collectionpickle = open(collection_file_str, "r",encoding="utf8")
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
        if  docalldata == None or docalldata[0] == None or len(docalldata[0]) <= 0:
           break    
        print("Doc Id : "+str(docalldata[0]+" count is   "+str(doccount)))
        curaddress = collectionpickle.tell()
    except :
 
        break


# In[ ]:


#query_file_str = "queries.doctrain.tsv"

queries = {}
with open(query_file_str,encoding="utf8") as fd:
    rd = csv.reader(fd, delimiter="\t", quoting = csv.QUOTE_NONE)
    print("Processing queries")
    for row in rd:
        qno = row[0]
        print("Query number : "+str(qno))
        quer = tokenizer.tokenize(row[1])
        quer = [ps.stem(w.lower()) for w in quer if (not w.lower() in stop_words and w.lower().isalnum()) ]
        qworddict = {}
        for w in quer:
            if w not in qworddict:
                qworddict[w] = 1 
            else :
                qworddict[w] += 1
        queries[qno] = qworddict
        
    


# In[32]:


#top_100_file_str = "msmarco-doctrain-top100"
uniquetops = []
querytops = {}
with open(top_100_file_str,encoding="utf8") as ftop100:
    print("Storing top 100")
    rtop100 = csv.reader(ftop100, delimiter = " ", quoting = csv.QUOTE_NONE)
    count = 0 
    for row in rtop100:
        
        count += 1
        if count % 100 == 1:
            curqno = row[0]
            print("New query : "+str(curqno))
            toparr = []
        toparr.append(row[2])
        if count % 100 == 0:
            querytops[curqno] = toparr
    
        #if count == 200 :
            #break
       


# In[33]:


#print(querytops)


# In[34]:


#len(querytops)


# In[35]:


#for query in querytops:
    #print(len(querytops[query]))


# In[ ]:




# In[ ]:


count = 0 

kappa = 0.9
k1 = 1.2
b = 0.75


# In[ ]:


outfile =  open("output_prob_rerank.txt", "w")
print("Training for queries")
for query in querytops:
    print("Taining for query : "+str(query))
    toparr = querytops[query]
    docs = {}
    for doc in toparr:
        collectionpickle.seek(docaddresses[doc])
        docalldata = collectionpickle.readline() 
        docalldata = docalldata.split(sep="\t")
        new_title = tokenizer.tokenize(docalldata[2])
        new_title = [ps.stem(w.lower()) for w in new_title if (not w.lower() in stop_words and w.lower().isalnum())]
        new_content = tokenizer.tokenize(docalldata[3])
        new_content = [ps.stem(w.lower()) for w in new_content if not w in stop_words]
        
        worddict = {}
        for w in new_title:
            if w not in worddict:
                worddict[w] = 1
            else:
                worddict[w] += 1
        for w in new_content:
            if w not in worddict:
                worddict[w] = 1
            else:
                worddict[w] += 1
        arr = []
        arr.append(worddict)
        arr.append(len(new_title) + len(new_content))
        docs[docalldata[0]] = arr
    lav = 0.0 
    for doc in docs:
        lav += docs[doc][1]
    lav = lav/len(docs)
    samqdict = queries[query]
    N = 100.0
    pis = {}
    uis = {}
    for termi in samqdict:
        dfi = 0.0
        for doc in docs:
            if termi in docs[doc][0]:
                dfi += 1
        pi = 1.0/3.0 + ((2.0/3.0)*dfi)/N
        pis[termi] = pi
        ui = dfi/N
        uis[termi] = ui
    weights = {}
    rsvs = {}
    for doc in docs:
        rsv = 0.0
        for termi in samqdict:
            wi = 0.0
            pi = pis[termi]
            ui = uis[termi]
            if (ui == 0.0) or ((1-pi) == 0.0) or ((pi/(1.0-pi))*((1.0-ui)/ui) <= 0.0):
                wi = 0.0 
            else:
                wi = 0.0
                rswi = math.log((pi/(1-pi))*((1-ui)/ui))
                qfi = samqdict[termi]
                tfi = 0.0
                if termi in docs[doc][0]:
                    tfi = docs[doc][0][termi]
                    l = docs[doc][1]
                    wi = (qfi*tfi*(1+k1)*rswi)/(k1*( (1-b) + (b*l)/lav ) + tfi)
            if doc not in weights:
                subdic = {}
                subdic[termi] = wi
                weights[doc] = subdic
            else:
                weights[doc][termi] = wi
            rsv += wi
        rsvs[doc] = rsv
    rsvs = dict( sorted(rsvs.items(), key=operator.itemgetter(1),reverse=True))
    
    # expansion
    numrel = 3
    pisnew = {}
    uisnew= {}
    for ecount in range(expansion_limit):
        print("current expansion :"+str(ecount+1))
        bpi = None
        bqi = None
        bestscore = None
        bestterm = None
        noterm = True
        reldocs = {}
        relc = 0 
        for doc in rsvs:
            if relc == numrel:
                break
            reldocs[doc] = rsvs[doc]
            relc += 1
        relterms = []
        for doc in reldocs:
            for term in docs[doc][0]:
                if term not in relterms:
                    relterms.append(term)
        for term in relterms:
            if term not in pisnew:
                dfi = 0.0
                for doc in docs:
                    if termi in docs[doc][0]:
                        dfi += 1
                pi = 1.0/3.0 + ((2.0/3.0)*dfi)/N
                pisnew[term] = pi
                uisnew[term] = dfi/N
            wi = 0.0
            pi = pisnew[term]
            ui = uisnew[term]
            rsvi = 0.0
            if (ui == 0.0) or ((1-pi) == 0.0) or ((pi/(1.0-pi))*((1.0-ui)/ui) <= 0.0):
                #print("hi")
                wi = 0.0 
                rsvi = 0.0

            else:
                wi = 0.0
                rsvi = math.log((pi/(1-pi))*((1-ui)/ui))
                if term in samqdict:
                    continue
                else:
                    qfi = 1
                tfi = 0.0
                if term in docs[doc][0]:
                    tfi = docs[doc][0][term]

                    l = docs[doc][1]
                    wi = (qfi*tfi*(1+k1)*rswi)/(k1*( (1-b) + (b*l)/lav ) + tfi)
                

            score =  rsvi*(pi-ui)
            if noterm == True:
                noterm = False
                bestscore = score
                bestterm = term
                bpi = pi
                bqi = ui
            else:
                if score > bestscore:
                    bestterm = term
                    bestscore = score
                    bpi = pi
                    bqi = ui
        if noterm == True :
            continue
        vr = len(reldocs)
        for termi in samqdict:
            vri = 0.0
            for reldoc in reldocs:
                if termi in docs[reldoc][0]:
                    vri += 1
            pis[termi] = (vri + kappa*pis[termi])/(vr + kappa)
            uis[termi] = (vri + kappa*uis[termi])/(vr + kappa)


        samqdict[bestterm] = 1
        pis[bestterm] = bpi
        uis[bestterm] = bqi


        for doc in docs:
            rsv = 0.0
            for termi in samqdict:
                wi = 0.0
                pi = pis[termi]
                ui = uis[termi]
                if (ui == 0.0) or ((1-pi) == 0.0) or ((pi/(1.0-pi))*((1.0-ui)/ui) <= 0.0):
                    wi = 0.0 
                else:
                    wi = 0.0
                    rswi = math.log((pi/(1-pi))*((1-ui)/ui))
                    qfi = samqdict[termi]
                    tfi = 0.0
                    if termi in docs[doc][0]:
                        tfi = docs[doc][0][termi]
                        l = docs[doc][1]
                        wi = (qfi*tfi*(1+k1)*rswi)/(k1*( (1-b) + (b*l)/lav ) + tfi)

                if doc not in weights:
                    subdic = {}
                    subdic[termi] = wi
                    weights[doc] = subdic
                else:
                    weights[doc][termi] = wi
                rsv += wi
            rsvs[doc] = rsv
        rsvs = dict( sorted(rsvs.items(), key=operator.itemgetter(1),reverse=True))
    
    curqno = query
    curank = 1
    for doc in rsvs:
        #print(str(curqno) + " Q0 " + str(doc) + " " + str(curank) + " " + str(rsvs[doc]) + " STANDARD\n" )
        outfile.write(str(curqno) + " Q0 " + str(doc) + " " + str(curank) + " " + str(rsvs[doc]) + " STANDARD\n"    )
        curank += 1

       

