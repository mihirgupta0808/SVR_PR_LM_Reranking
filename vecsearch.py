k = 10
import sys
import math
import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk.stem import PorterStemmer
import pickle
import heapdict

long_options = ["query","cutoff","output","index","dict"]
full_args = sys.argv
argument_list = full_args[1:]
#print(argument_list)
argkey= ""
argdict = {}
key = ""
for word in argument_list:
    if word[0:2] == "--":
        key = word
    else:
        argdict[key] = word
#print("argdict")
#print(argdict)

for current_argument in argdict.keys():
    #print(current_argument)
    current_value = argdict[current_argument]
    #print(current_value)
    if current_argument == "--query":
        qfilestr = current_value
        #print("qfile:" + qfilestr)
    elif current_argument == "--cutoff":
        k = int(current_value)
    elif current_argument == "--output":
        outstr = current_value
    elif current_argument == "--index":
        indexstr = current_value
    elif current_argument == "--dict":
        dictstr = current_value

idxfileread = open(indexstr,"rb")
dictfileread = open(dictstr, "rb")

N = pickle.load(dictfileread)

#print(N)

dictall = {}
while True:
    try:
        dictdata = pickle.load(dictfileread)
        darr = []
        darr.append(dictdata[1])
        darr.append(dictdata[2])
        darr.append(dictdata[3])
        dictall[dictdata[0]] = darr
        #print(dictdata[0] + ":" + str(dictdata[1]) + ":" + str(dictdata[2])  + ":" + dictdata[3] )

    except EOFError:
        break
idfs = {}

for key in dictall.keys():
    #print(dictall[key][0])
    idfs[key] = math.log2(float(N/dictall[key][0]))
    #print(idfs[key])

tfs = {}
# tfs is a dictionary that will store keys as documents(j) and and then this key will store a dictionary haveing keys as word(i) and the value as tfij
# thus tfs will store all document vectors 
for key in dictall:
    #print(dictall[key][1])
    idxfileread.seek(dictall[key][1])
    idxdata = pickle.load(idxfileread)
    for doc in idxdata:
        if doc not in tfs:
            keydict = {}
            tfij = 1 + math.log2(idxdata[doc])
            keydict[key] = tfij
            tfs[doc] = keydict
        else:
            keydict = tfs[doc]
            tfij = 1 + math.log2(idxdata[doc])
            keydict[key] = tfij
            tfs[doc] = keydict
                
    #print(key)
    #print(idxdata)
moddocs = {}
for doc in tfs.keys():
    moddoc = 0.0
    for word in tfs[doc]:
        moddoc += (tfs[doc][word]*idfs[word])**2
    moddoc = math.sqrt(moddoc)
    moddocs[doc] = moddoc
#print(moddocs)
ps = PorterStemmer()
dicstems = {}
stems = {}
for key in dictall.keys():
    stem = ps.stem(key)
    dicstems[key] = stem
    if stem not in stems:
        starr = []
        starr.append(key)
        stems[stem] = starr
    else:
        starr = stems[stem]
        if key not in starr:
            starr.append(key)
            stems[stem] = starr
            
#print(dicstems)
#print(stems)
jar = "stanford-ner-4.0.0.jar"
model = "english.all.3class.distsim.crf.ser.gz"
stan = StanfordNERTagger(model, jar, encoding='utf8')
class TrieNode(): 
    def __init__(self): 
        self.children = {} 
        self.end = False
  
class Trie(): 
    def __init__(self): 
        self.root = TrieNode() 
   
    def insert(self, key): 
        node = self.root 
        for ch in list(key): 
            if not node.children.get(ch): 
                node.children[ch] = TrieNode() 
  
            node = node.children[ch] 
  
        node.end = True
  

    def dfs(self,node,word,wordlist):  
        if node.end: 
            wordlist.append(word) 
  
        for ch,nextnode in node.children.items(): 
            self.dfs(nextnode, word + ch,wordlist) 
  
    def querySearch(self, key):
        wordlist = []
        node = self.root 
        not_found = False
        word = "" 
        for ch in key: 
            if not node.children.get(ch): 
                return wordlist
            word += ch 
            node = node.children[ch] 
   
        self.dfs(node,word,wordlist) 
    
        return wordlist
prefixsearch = Trie()
for key in dictall.keys():
    prefixsearch.insert(key)







outfile =  open(outstr, "w")
qfile = open(qfilestr, 'r') 
Lines = qfile.readlines() 
wc = 0
count = 0
qno = 0 
firstq = False

for line in Lines: 
    lstrip = line.strip().lower().split()
    strip2 = line.strip().split()
    #print(lstrip)
    if len(lstrip) > 0 and lstrip[0] == "<num>":
        if firstq == False:
            firstq = True
        else:
            outfile.write("\n")
            outfile.write("\n")
        
        #print(lstrip)
        qno = lstrip[2]
        #print(qno)
        if qno.isdigit():
            qno = int(qno)
        #print(qno)
    if len(lstrip) > 0 and lstrip[0] == "<title>":
        count += 1
        del(lstrip[0])
        del(lstrip[0])
        del(strip2[0])
        del(strip2[0])
        newline = ""
        for word in strip2:
            newline+= word 
            newline+= " "
        newline.strip()
        tokens = nltk.word_tokenize(newline) 
        tagged = stan.tag(tokens)
        tagdic = {}
        for tup in tagged:
            tagdic[tup[0].lower()] = tup[1]
        #print(tagged)
        #print(tagdic)
        qstems = {}
        for q in tagdic.keys():
            qstems[q] = ps.stem(q)
        #print(qstems)

            
        #print("query is :")
        #print(line)
        #print(count)
        #print(count)
        fcounts = {}
        tfq = {}
        nomatch = True
        for word in tagdic.keys():
            if word in idfs or qstems[word] in stems:
                matches = []
                if word in idfs:
                    #print(word + " matched normally: " + " dic tag: " + dictall[word][2]  )
                    matches.append(word)
                if qstems[word] in stems:
                    #print(" some stems matched")
                    revstems = stems[qstems[word]]
                    for revstem in revstems:
                        if revstem == word:
                            #print("stem of original: " + word  + "  matched ,dic tag :" + dictall[word][2] )
                            if revstem not in matches:
                                matches.append(revstem)
                        else:
                            #print(revstem + "matched: " + " dic tag : " + dictall[revstem][2])
                            matches.append(revstem)
                if "*" in word:
                    ind = word.find("*")
                    if ind == -1 :
                        continue
                    qword = word[0:ind]
                    prefmatches = prefixsearch.querySearch(qword)
                    for prefmatch in prefmatches:
                        if prefmatch not in matches:
                            matches.append(prefmatch)
                    
                
                            
                    
                    
                        
                
                #print("now checking matching of tags")
                wc += 1
                #print(wc)
                for match in matches:
                    qtag = tagdic[word]
                    dtag = dictall[match][2]
                    b1 = (qtag == "O" )
                    b2 = (qtag == "ORGANIZATION" and dtag == "o")
                    b3 = (qtag == "LOCATION" and dtag == "l")
                    b4 = (qtag == "PERSON" and dtag == "p")
                    if (b1 or b2 or b3 or b4):
                        
                        #print("tags of : " + match + " , matched as well ")
                        nomatch = False
                        if match not in fcounts:
                            fcounts[match] = 1
                        else:
                            fcounts[match] += 1
                    #else:
                        #print("tags of: " + match + " , didn't match")
            #print(word)
        if nomatch == False:
            modq = 0.0
            for word in fcounts.keys():
                tfq[word] = 1 + math.log2(fcounts[word])
                modq += (tfq[word]*idfs[word])**2
            modq = math.sqrt(modq)

        h = heapdict.heapdict() 
        for doc in tfs.keys():
            costheta = 0.0
            #print(doc)
            dotp = 0.0 
            if nomatch == True:
                costheta = 0.0
                
            else:
                
                for word in tfq.keys():
                    wq = idfs[word]*tfq[word]
                    if doc not in tfs or word not in tfs[doc]:
                        wdoc = 0.0 
                    else:
                        wdoc = tfs[doc][word]*idfs[word]
                    dotp += wq*wdoc
            #costheta = (dotp/modq)

                moddoc = moddocs[doc]


                costheta = (dotp/moddoc)/modq
            #if costheta != 0 :
                #print(doc)
                #print("with " + doc + "costheta is:" + str(costheta))
            
            
            h[doc] = -costheta
        for i in range(k):
            hitem = h.peekitem()
            h.popitem()
            #print( str(i+1) + ":" + hitem[0] + ":" + str(-hitem[1]))
            # i+1 instead of 0
            outfile.write(str(qno) + " Q0 " + hitem[0] + " " + str(i+1) + " " + str(-hitem[1]) + " STANDARD\n"    )
            
            
        
qfile.close()
outfile.close()
#print("end")
            
            
