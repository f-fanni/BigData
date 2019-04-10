from testPipe import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import *
from pyspark.ml.feature import *

def getSpark():
    return SparkSession.builder.getOrCreate()

spark = getSpark()

index = 8

lr = LogisticRegression(regParam=0.1, maxIter=20)
pipeline = Pipeline(stages=[lr])
description = "Logistic Regression con regParam=0.1 e maxIter=20"
testPipeline(pipeline, description, index)

lr = LogisticRegression(regParam=0.01, maxIter=20)
pipeline = Pipeline(stages=[lr])
description = "Logistic Regression con regParam=0.01 e maxIter=20"
testPipeline(pipeline, description, index)

spark.stop()
spark = getSpark()

#RANDOM FOREST
lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
pipeline = Pipeline(stages=[lr])
description = "Random Forest con maxDepth=5 e numTrees=20"
testPipeline(pipeline, description, index)

lr = RandomForestClassifier(maxDepth=8, numTrees=40, seed=42)
pipeline = Pipeline(stages=[lr])
description = "Random Forest con maxDepth=8 e numTrees=40"
testPipeline(pipeline, description, index)

spark.stop()
spark = getSpark()

index = 9

lr = LogisticRegression(regParam=0.1, maxIter=20)
pipeline = Pipeline(stages=[lr])
description = "Logistic Regression con regParam=0.1 e maxIter=20"
testPipeline(pipeline, description, index)

lr = LogisticRegression(regParam=0.01, maxIter=20)
pipeline = Pipeline(stages=[lr])
description = "Logistic Regression con regParam=0.01 e maxIter=20"
testPipeline(pipeline, description, index)

spark.stop()
spark = getSpark()
#RANDOM FOREST
lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
pipeline = Pipeline(stages=[lr])
description = "Random Forest con maxDepth=5 e numTrees=20"
testPipeline(pipeline, description, index)

lr = RandomForestClassifier(maxDepth=8, numTrees=40, seed=42)
pipeline = Pipeline(stages=[lr])
description = "Random Forest con maxDepth=8 e numTrees=40"
testPipeline(pipeline, description, index)


spark.stop()
spark = getSpark()

index = 10

lr = LogisticRegression(regParam=0.1, maxIter=20)
pipeline = Pipeline(stages=[lr])
description = "Logistic Regression con regParam=0.1 e maxIter=20"
testPipeline(pipeline, description, index)

lr = LogisticRegression(regParam=0.01, maxIter=20)
pipeline = Pipeline(stages=[lr])
description = "Logistic Regression con regParam=0.01 e maxIter=20"
testPipeline(pipeline, description, index)

spark.stop()
spark = getSpark()
#RANDOM FOREST
lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
pipeline = Pipeline(stages=[lr])
description = "Random Forest con maxDepth=5 e numTrees=20"
testPipeline(pipeline, description, index)

lr = RandomForestClassifier(maxDepth=8, numTrees=40, seed=42)
pipeline = Pipeline(stages=[lr])
description = "Random Forest con maxDepth=8 e numTrees=40"
testPipeline(pipeline, description, index)
