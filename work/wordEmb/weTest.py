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

spark = SparkSession.builder.getOrCreate()
print(spark.sparkContext.defaultParallelism)
df = spark.read.load('../../dataset/merged/publisher/')
df = df.withColumn('label', df._hyperpartisan.cast('integer'))
testSet = spark.read.load('../../dataset/merged/article/')
testSet = testSet.withColumn('label', testSet._hyperpartisan.cast('integer'))

filename = "../../dataset/wordembeddings/embeddings_snap_s512_e15.txt_m"
index = 8

wordDict = spark.sparkContext.broadcast({row[0] : np.asarray(row[1:]) for _, row in pd.read_csv(filename, skiprows=1, delim_whitespace=True, encoding='latin').iterrows()})

def embedder(words):
    aux = []
    for word in words:
        try:
            aux.append(wordDict.value[word])
        except:
            pass
    if(len(aux) == 0):
        aux = [np.zeros(shape=(len(wordDict.value['house'])))]
    return Vectors.dense(np.mean(aux, axis=0, dtype=np.float64))

embedder_udf = udf(embedder, VectorUDT())
remover = StopWordsRemover(inputCol="words", outputCol="filtered")


df1 = remover.transform(df).drop('words')
df1 = df1.withColumn("features", embedder_udf("filtered")).drop('filtered')
df2 = df.withColumn("features", embedder_udf("words")).drop('words')
testSet1 = remover.transform(testSet).drop('words')
testSet1 = testSet1.withColumn("features", embedder_udf("filtered")).drop('filtered')
testSet2 = testSet.withColumn("features", embedder_udf("words")).drop('words')

df1.write.save(f'publisherFiltered{index}', format='parquet', mode="error")
testSet1.write.save(f'articleFiltered{index}', format='parquet', mode="error")
df2.write.save(f'publisherRaw{index}', format='parquet', mode="error")
testSet2.write.save(f'articleRaw{index}', format='parquet', mode="error")

df = spark.read.load('../../dataset/merged/publisher/')
df = df.withColumn('label', df._hyperpartisan.cast('integer'))
testSet = spark.read.load('../../dataset/merged/article/')
testSet = testSet.withColumn('label', testSet._hyperpartisan.cast('integer'))



filename = "../../dataset/wordembeddings/embeddings_snap_s512_e30.txt_m"
index = 9

wordDict = spark.sparkContext.broadcast({row[0] : np.asarray(row[1:]) for _, row in pd.read_csv(filename, skiprows=1, delim_whitespace=True, encoding='latin').iterrows()})

def embedder(words):
    aux = []
    for word in words:
        try:
            aux.append(wordDict.value[word])
        except:
            pass
    if(len(aux) == 0):
        aux = [np.zeros(shape=(len(wordDict.value['house'])))]
    return Vectors.dense(np.mean(aux, axis=0, dtype=np.float64))

embedder_udf = udf(embedder, VectorUDT())
remover = StopWordsRemover(inputCol="words", outputCol="filtered")


df1 = remover.transform(df).drop('words')
df1 = df1.withColumn("features", embedder_udf("filtered")).drop('filtered')
df2 = df.withColumn("features", embedder_udf("words")).drop('words')
testSet1 = remover.transform(testSet).drop('words')
testSet1 = testSet1.withColumn("features", embedder_udf("filtered")).drop('filtered')
testSet2 = testSet.withColumn("features", embedder_udf("words")).drop('words')

df1.write.save(f'publisherFiltered{index}', format='parquet', mode="error")
testSet1.write.save(f'articleFiltered{index}', format='parquet', mode="error")
df2.write.save(f'publisherRaw{index}', format='parquet', mode="error")
testSet2.write.save(f'articleRaw{index}', format='parquet', mode="error")

df = spark.read.load('../../dataset/merged/publisher/')
df = df.withColumn('label', df._hyperpartisan.cast('integer'))
testSet = spark.read.load('../../dataset/merged/article/')
testSet = testSet.withColumn('label', testSet._hyperpartisan.cast('integer'))


filename = "../../dataset/wordembeddings/embeddings_snap_s512_e50.txt_m"
index = 10

wordDict = spark.sparkContext.broadcast({row[0] : np.asarray(row[1:]) for _, row in pd.read_csv(filename, skiprows=1, delim_whitespace=True, encoding='latin').iterrows()})

def embedder(words):
    aux = []
    for word in words:
        try:
            aux.append(wordDict.value[word])
        except:
            pass
    if(len(aux) == 0):
        aux = [np.zeros(shape=(len(wordDict.value['house'])))]
    return Vectors.dense(np.mean(aux, axis=0, dtype=np.float64))

embedder_udf = udf(embedder, VectorUDT())
remover = StopWordsRemover(inputCol="words", outputCol="filtered")


df1 = remover.transform(df).drop('words')
df1 = df1.withColumn("features", embedder_udf("filtered")).drop('filtered')
df2 = df.withColumn("features", embedder_udf("words")).drop('words')
testSet1 = remover.transform(testSet).drop('words')
testSet1 = testSet1.withColumn("features", embedder_udf("filtered")).drop('filtered')
testSet2 = testSet.withColumn("features", embedder_udf("words")).drop('words')

df1.write.save(f'publisherFiltered{index}', format='parquet', mode="error")
testSet1.write.save(f'articleFiltered{index}', format='parquet', mode="error")
df2.write.save(f'publisherRaw{index}', format='parquet', mode="error")
testSet2.write.save(f'articleRaw{index}', format='parquet', mode="error")