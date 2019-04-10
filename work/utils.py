from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import itertools
import numpy as np
from pyspark import since, keyword_only
from pyspark.ml import Estimator, Model
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.param.shared import HasSeed
from pyspark.sql.functions import rand
from pyspark.ml.tuning import *


def getSpark():
    return SparkSession.builder.getOrCreate()


def validateModel(model, filename):
    test = spark.read.load('../dataset/merged/article/')
    test = test.withColumn('label', test._hyperpartisan.cast('integer'))
    test = model.transform(test)
    ev = BinaryClassificationEvaluator()
    with open(filename,"a") as file:
        file.write(f"{ev.getMetricName()}: {ev.evaluate(test)}\n")
        ev = MulticlassClassificationEvaluator()
        file.write(f"{ev.getMetricName()}: {ev.evaluate(test)}\n")
        ev.setMetricName("weightedPrecision")
        file.write(f"{ev.getMetricName()}: {ev.evaluate(test)}\n")
        ev.setMetricName("weightedRecall")
        file.write(f"{ev.getMetricName()}: {ev.evaluate(test)}\n")
        ev.setMetricName("accuracy")
        file.write(f"{ev.getMetricName()}: {ev.evaluate(test)}\n")


class Tuner(TrainValidationSplit):
    def fit(self, train, validation):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        metrics = [0.0] * numModels
        models = est.fit(train, epm)
        for j in range(numModels):
            model = models[j]
            metric = eva.evaluate(model.transform(validation, epm[j]))
            metrics[j] += metric
        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(train.union(validation), epm[bestIndex])
        return self._copyValues(TrainValidationSplitModel(bestModel, metrics))