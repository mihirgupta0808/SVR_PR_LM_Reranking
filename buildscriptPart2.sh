#!/bin/bash


pip install nltk

if [ $? -eq 0 ]
then
    echo "Successfully installed nltk"
else
    echo "Could not install nltk"
fi

python prob_rerank.py ~/scratch/MSMARCO-DocRanking/msmarco-docdev-queries.tsv ~/scratch/MSMARCO-DocRanking/msmarco-docdev-top100  ~/scratch/MSMARCO-DocRanking/msmarco-docs.tsv 10

python lm_rerank.py ~/scratch/MSMARCO-DocRanking/msmarco-docdev-queries.tsv ~/scratch/MSMARCO-DocRanking/msmarco-docdev-top100  ~/scratch/MSMARCO-DocRanking/msmarco-docs.tsv uni

python lm_rerank.py ~/scratch/MSMARCO-DocRanking/msmarco-docdev-queries.tsv ~/scratch/MSMARCO-DocRanking/msmarco-docdev-top100  ~/scratch/MSMARCO-DocRanking/msmarco-docs.tsv bi

