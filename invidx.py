import sys
from bs4 import BeautifulSoup
import os
import pickle
import time
import string
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
#print(stop_words)

inverted_list = {}
labels = {}
weirds = ["''","'d'","'s","-lrb-","-rrb-","``"]


#print (time.asctime( time.localtime(time.time()) ))

count = 0
N = 0
#folder = os.path.join(os.getcwd(),"traindata")
folder = sys.argv[1]
#print(folder)
for filename in os.listdir(folder):
    #print("going through : " + filename)

    infile = open(os.path.join(folder,filename),"r")
    
    
    contents = infile.read()
    contents = "<BEGIN>" + contents + "</BEGIN>"
    soup = BeautifulSoup(contents, 'xml')
        
    docs = soup.find_all('DOC')
    
    
    for doc in docs:
        N += 1
        docId = doc.DOCNO.contents[0]
        #docId.replace(" ", "")
        docId = docId.strip()
        #print(docId)
        #print(len(docId))
        text = doc.find_all('TEXT')

        for tx in text:
            if(len(tx.contents) == 0):
                continue
            #print(tx)
            locs = tx.find_all('LOCATION')
            orgs = tx.find_all('ORGANIZATION')
            pers = tx.find_all('PERSON')
            #print(locs)
            for loc in locs:
                wstr = loc.contents[0].lower()
                cstr = wstr.strip()
               
                #print(cstr)
                # "l" 3rd arg for location
                if len(cstr) == 0:
                    continue
                if cstr in string.punctuation:
                    continue
                if cstr in stop_words:
                    continue
                #print(cstr)
                #print(len(cstr))
                if cstr in weirds:
                    continue
                if cstr not in inverted_list:
                    labels[cstr] = "l"
                    dic = {}
                    dic[docId] = 1
                    inverted_list[cstr] = dic 
                else:
                    if docId in inverted_list[cstr]:
                        inverted_list[cstr][docId] += 1
                    else:
                        dic = inverted_list[cstr]
                        dic[docId] = 1
                        inverted_list[cstr] = dic
            for org in orgs:

                wstr = org.contents[0].lower()
                #cstr = str(wstr.string)
                cstr = wstr.strip()
                    
                if len(cstr) == 0 :
                    continue
                if cstr in string.punctuation:
                    continue
                if cstr in stop_words:
                    continue
                #print(cstr)
                if cstr in weirds:
                    continue
                if cstr not in inverted_list:
                    labels[cstr] = "o"
                    dic = {}
                    dic[docId] = 1
                    inverted_list[cstr] = dic 
                else:
                    if docId in inverted_list[cstr]:
                        inverted_list[cstr][docId] += 1
                    else:
                        dic = inverted_list[cstr]
                        dic[docId] = 1
                        inverted_list[cstr] = dic
            for per in pers:
                wstr = per.contents[0].lower()
                cstr = wstr.strip()
                if len(cstr) == 0 :
                    continue
                if cstr in string.punctuation:
                    continue
                if cstr in weirds:
                    continue
                
                if cstr in stop_words:
                    continue
                #print(cstr)
                if cstr not in inverted_list:
                    labels[cstr] = "p"
                    dic = {}
                    dic[docId] = 1
                    inverted_list[cstr] = dic 
                else:
                    if docId in inverted_list[cstr]:
                        inverted_list[cstr][docId] += 1
                    else:
                        dic = inverted_list[cstr]
                        dic[docId] = 1
                        inverted_list[cstr] = dic
    infile.close()
    count+=1
    #print(count)
    #if(count == 10):
        #break
sorted_dict = dict(sorted(inverted_list.items()))
#print(sorted_dict)
rootfileword = sys.argv[2]
#dictfile = open("indexfile.dict","wb")
dictfile = open(rootfileword +".dict","wb")
#idxfile = open("indexfile.idx","wb")
idxfile = open(rootfileword +".idx","wb")
pickle.dump(N,dictfile)
for key in sorted_dict.keys():
    arr = []
    arr.append(key)
    val = 0 
    val = len(sorted_dict[key])
    arr.append(val)
    arr.append(idxfile.tell())
    arr.append(labels[key])
    pickle.dump(arr,dictfile)
    pickle.dump(sorted_dict[key],idxfile)

#print (time.asctime( time.localtime(time.time()) ))









