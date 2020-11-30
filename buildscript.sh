#!/bin/bash

echo "hello mere yaaar 1"

pip install bs4

echo "hello mer yaar 2"
if [ $? -eq 0 ]
then
    echo "Successfully installed bs4"
else
    echo "Could not install bs4"
fi

echo "hello mer yaar 3"

pip install lxml
if [ $? -eq 0 ]
then
    echo "Successfully installed lxml"
else
    echo "Could not install lxml"
fi

pip install heapdict

if [ $? -eq 0 ]
then
    echo "Successfully installed heapdict"
else
    echo "Could not install heapdict"
fi

pip install nltk

if [ $? -eq 0 ]
then
    echo "Successfully installed nltk"
else
    echo "Could not install nltk"
fi

# It is assumed in the code that the files "stanford-ner-4.0.0.jar", 
#"english.all.3class.distsim.crf.ser.gz" for 
#using the StanfordNERTagger are present in the same directory as vecsearch.py . 
# Please see if they are present in the current directory for vecsearch.py

#python invidx.py ./TrainingData indexfile

# coll-path is assumed to be location of training data 
# and indexfile is outputname of .idx and .dict files
if [ $? -eq 0 ]
then
    echo "Successfully ran invidx.py "
else
    echo "Could not run invidx.py"
fi

python printdict.py indexfile.dict

if [ $? -eq 0 ]
then
    echo "Successfully ran printdict.py"
else
    echo "Could not run printdict.py"
fi

python vecsearch.py --query qrels.filtered.51-100 --cutoff 10 --output resultfile --index indexfile.idx --dict indexfile.dict
# queryfile is assumed to be the file containing queries
#resultfile is the file in which results will be generated, indexfile.idx is the.idx file 
# indexfile.dict is the .dictfile
if [ $? -eq 0 ]
then
    echo "Successfully ran vecsearch.py"
else
    echo "Could not run vecsearch.py"
fi


