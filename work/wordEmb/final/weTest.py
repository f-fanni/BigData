from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, lit, udf
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer
import string
import re
import csv
from pyspark.ml import Pipeline
from pyspark.ml.classification import *
from pyspark.ml.feature import *
from pyspark.ml.tuning import *
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pickle
import pandas as pd
import numpy as np
from pyspark.ml.linalg import Vectors, VectorUDT


index2filename = {
    1: '../../../dataset/wordembeddings/sentic2vec.csv',
    2: '../../../dataset/wordembeddings/embeddings_snap_s128_e15.txt_m',
    3: '../../../dataset/wordembeddings/embeddings_snap_s128_e30.txt_m',
    4: '../../../dataset/wordembeddings/embeddings_snap_s128_e50.txt_m',
    5: '../../../dataset/wordembeddings/embeddings_snap_s256_e15.txt_m',
    6: '../../../dataset/wordembeddings/embeddings_snap_s256_e30.txt_m',
    7: '../../../dataset/wordembeddings/embeddings_snap_s256_e50.txt_m',
    8: '../../../dataset/wordembeddings/embeddings_snap_s512_e15.txt_m',
    9: '../../../dataset/wordembeddings/embeddings_snap_s512_e30.txt_m',
    10: '../../../dataset/wordembeddings/embeddings_snap_s512_e50.txt_m',
}

def wordEmbedder(indices):

    for index in indices:
        filename = index2filename[index]

        spark = SparkSession.builder.master("local[8]").getOrCreate()
        df = spark.read.load('../../../dataset/merged/publisher/')
        df = df.withColumn('label', df._hyperpartisan.cast('integer'))
        testSet = spark.read.load('../../../dataset/merged/article/')
        testSet = testSet.withColumn('label', testSet._hyperpartisan.cast('integer'))

        if(inedx == 1):
            wordDict = spark.sparkContext.broadcast({row[0] : np.asarray(row[1:]) for _, row in pd.read_csv(filename, skiprows=1, encoding='latin').iterrows()})
        else:
            wordDict = spark.sparkContext.broadcast({row[0] : np.asarray(row[1:]) for _, row in pd.read_csv(filename, skiprows=1, delim_whitespace=True, encoding='latin').iterrows()})

        def embedder(words):
            aux = []
            for word in words:
                try:
                    aux.append(wordDict.value[word])
                except:
                    pass
            if(len(aux) == 0):
                return None
            #modifiy the return to test alternative strategies, like avg, sum, etc.
            return Vectors.dense(np.concatenate((np.sum(aux, axis=0, dtype=np.float64), np.std(aux, axis=0, dtype=np.float64))))

        embedder_udf = udf(embedder, VectorUDT())
        remover = StopWordsRemover(inputCol="words", outputCol="filtered")


        df1 = remover.transform(df).drop('words')
        df1 = df1.withColumn("features", embedder_udf("filtered")).drop('filtered').dropna(subset=['features'])
        df2 = df.withColumn("features", embedder_udf("words")).drop('words').dropna(subset=['features'])
        testSet1 = remover.transform(testSet).drop('words')
        testSet1 = testSet1.withColumn("features", embedder_udf("filtered")).drop('filtered')
        testSet2 = testSet.withColumn("features", embedder_udf("words")).drop('words')

        df1.write.save(f'publisherFiltered{index}', format='parquet', mode="error")
        testSet1.write.save(f'articleFiltered{index}', format='parquet', mode="error")
        df2.write.save(f'publisherRaw{index}', format='parquet', mode="error")
        testSet2.write.save(f'articleRaw{index}', format='parquet', mode="error")

        spark.stop()