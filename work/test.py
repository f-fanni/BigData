from utils import *
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


spark = getSpark()

df = spark.read.load('../dataset/merged/publisher/')
df = df.withColumn('label', df._hyperpartisan.cast('integer'))
testSet = spark.read.load('../dataset/merged/article/')
testSet = testSet.withColumn('label', testSet._hyperpartisan.cast('integer'))

hashingTF = HashingTF(inputCol="words", outputCol="rawfeatures", numFeatures=1000)
idf = IDF(inputCol="rawfeatures", outputCol="features")
#pca = PCA(k=1000, inputCol="rfeatures", outputCol="features")
#lr = LogisticRegression(regParam=0.1, maxIter=20)
lr = RandomForestClassifier(numTrees=20, maxDepth=5, seed=42)
pipeline = Pipeline(stages=[hashingTF, idf, lr])
filename = "TFIDF-RF20-5"

ev = MulticlassClassificationEvaluator(metricName='accuracy')
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=ParamGridBuilder().build(),
                          evaluator=ev,
                          numFolds=10, seed=42)

model = crossval.fit(df)

with open(filename,"a") as f:
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
