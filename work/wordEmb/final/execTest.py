from testPipe import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import *
from pyspark.ml.feature import *

def getSpark():
    return SparkSession.builder.master("local[8]").getOrCreate()

def multiTest(indices):
    spark = getSpark()

    for index in indices:

        spark.stop()
        spark = getSpark()

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
        

def multiTest2(indices):
    spark = getSpark()

    for index in indices:
        '''
        spark.stop()
        spark = getSpark()

        lr = LogisticRegression(regParam=0.01, maxIter=50)
        pipeline = Pipeline(stages=[lr])
        description = "Logistic Regression con regParam=0.01 e maxIter=50"
        testPipeline(pipeline, description, index)

        lr = LogisticRegression(regParam=0.01, maxIter=100)
        pipeline = Pipeline(stages=[lr])
        description = "Logistic Regression con regParam=0.01 e maxIter=100"
        testPipeline(pipeline, description, index)

        spark.stop()
        spark = getSpark()

        #RANDOM FOREST
        lr = RandomForestClassifier(maxDepth=10, numTrees=10, seed=42)
        pipeline = Pipeline(stages=[lr])
        description = "Random Forest con maxDepth=10 e numTrees=10"
        testPipeline(pipeline, description, index)

        lr = RandomForestClassifier(maxDepth=10, numTrees=20, seed=42)
        pipeline = Pipeline(stages=[lr])
        description = "Random Forest con maxDepth=10 e numTrees=20"
        testPipeline(pipeline, description, index)
        
        lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
        pipeline = Pipeline(stages=[lr])
        description = "Random Forest con maxDepth=5 e numTrees=20"
        testPipeline(pipeline, description, index)
        
        lr = RandomForestClassifier(maxDepth=5, numTrees=80, seed=42)
        pipeline = Pipeline(stages=[lr])
        description = "Random Forest con maxDepth=5 e numTrees=80"
        testPipeline(pipeline, description, index)
        '''
        lr = RandomForestClassifier(maxDepth=10, numTrees=100, seed=42)
        pipeline = Pipeline(stages=[lr])
        description = "Random Forest con maxDepth=10 e numTrees=100"
        testPipeline(pipeline, description, index)