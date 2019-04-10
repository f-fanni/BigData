from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, lit, udf
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer
import string
import re
from pyspark.ml import Pipeline
from pyspark.ml.classification import *
from pyspark.ml.feature import HashingTF, Tokenizer, IDF, PCA
from pyspark.ml.tuning import *
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pickle

def getSpark():
    return SparkSession.builder.getOrCreate()


def testPipeline(pipeline, description, index):
    spark = getSpark()
####Raw
    df = spark.read.load(f'publisherRaw{index}/')
    df = df.withColumn('label', df._hyperpartisan.cast('integer'))
    testSet = spark.read.load(f'articleRaw{index}/')
    testSet = testSet.withColumn('label', testSet._hyperpartisan.cast('integer'))

    filename = f"resultsRaw{index}"

    ev = MulticlassClassificationEvaluator(metricName='accuracy')
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=ParamGridBuilder().build(),
                              evaluator=ev,
                              numFolds=10, seed=42)

    model = crossval.fit(df)

    with open(filename,"a") as f:
        f.write(f"{description}\n")
        f.write(f"accuracy crossValidation: {max(model.avgMetrics)}\n")
        f.write(f"accuracy testSet        : {ev.evaluate(model.transform(testSet))}\n")


    ev = MulticlassClassificationEvaluator(metricName='weightedPrecision')
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=ParamGridBuilder().build(),
                              evaluator=ev,
                              numFolds=10, seed=42)

    model = crossval.fit(df)


    with open(filename,"a") as f:
        f.write(f"precision crossValidation: {max(model.avgMetrics)}\n")
        f.write(f"precision testSet        : {ev.evaluate(model.transform(testSet))}\n")


    ev = MulticlassClassificationEvaluator(metricName='weightedRecall')
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=ParamGridBuilder().build(),
                              evaluator=ev,
                              numFolds=10, seed=42)

    model = crossval.fit(df)


    with open(filename,"a") as f:
        f.write(f"recall crossValidation: {max(model.avgMetrics)}\n")
        f.write(f"recall testSet        : {ev.evaluate(model.transform(testSet))}\n")

        
    ev = MulticlassClassificationEvaluator(metricName='f1')
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=ParamGridBuilder().build(),
                              evaluator=ev,
                              numFolds=10, seed=42)

    model = crossval.fit(df)


    with open(filename,"a") as f:
        f.write(f"f1 crossValidation: {max(model.avgMetrics)}\n")
        f.write(f"f1 testSet        : {ev.evaluate(model.transform(testSet))}\n\n\n")

###########Filtered        
    df = spark.read.load(f'publisherFiltered{index}/')
    df = df.withColumn('label', df._hyperpartisan.cast('integer'))
    testSet = spark.read.load(f'articleFiltered{index}/')
    testSet = testSet.withColumn('label', testSet._hyperpartisan.cast('integer'))

    filename = f"resultsFiltered{index}"

    ev = MulticlassClassificationEvaluator(metricName='accuracy')
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=ParamGridBuilder().build(),
                              evaluator=ev,
                              numFolds=10, seed=42)

    model = crossval.fit(df)

    with open(filename,"a") as f:
        f.write(f"{description}\n")
        f.write(f"accuracy crossValidation: {max(model.avgMetrics)}\n")
        f.write(f"accuracy testSet        : {ev.evaluate(model.transform(testSet))}\n")


    ev = MulticlassClassificationEvaluator(metricName='weightedPrecision')
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=ParamGridBuilder().build(),
                              evaluator=ev,
                              numFolds=10, seed=42)

    model = crossval.fit(df)


    with open(filename,"a") as f:
        f.write(f"precision crossValidation: {max(model.avgMetrics)}\n")
        f.write(f"precision testSet        : {ev.evaluate(model.transform(testSet))}\n")


    ev = MulticlassClassificationEvaluator(metricName='weightedRecall')
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=ParamGridBuilder().build(),
                              evaluator=ev,
                              numFolds=10, seed=42)

    model = crossval.fit(df)


    with open(filename,"a") as f:
        f.write(f"recall crossValidation: {max(model.avgMetrics)}\n")
        f.write(f"recall testSet        : {ev.evaluate(model.transform(testSet))}\n")

        
    ev = MulticlassClassificationEvaluator(metricName='f1')
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=ParamGridBuilder().build(),
                              evaluator=ev,
                              numFolds=10, seed=42)

    model = crossval.fit(df)


    with open(filename,"a") as f:
        f.write(f"f1 crossValidation: {max(model.avgMetrics)}\n")
        f.write(f"f1 testSet        : {ev.evaluate(model.transform(testSet))}\n\n\n")
