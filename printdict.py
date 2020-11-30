import pickle
import sys
filename = sys.argv[1]
dictfileread= open(filename, "rb")


N = pickle.load(dictfileread)
#print(N)

while True:
    try:
        dictdata = pickle.load(dictfileread)
        #darr = []
        #darr.append(dictdata[1])
        #darr.append(dictdata[2])
        #darr.append(dictdata[3])
        #dictall[dictdata[0]] = darr
        print(dictdata[0] + ":" + str(dictdata[1]) + ":" + str(dictdata[2]))
       
    except EOFError:
        break


