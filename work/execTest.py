from utils import *
from testPipeline import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import *
from pyspark.ml.feature import *

spark = getSpark()

hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=1000)
lr = LogisticRegression(regParam=0.1, maxIter=20)
pipeline = Pipeline(stages=[hashingTF, lr])
description = "Hashing Term Frequencies con 1000 features in output\nLogistic Regression con regParam=0.1 e maxIter=20"
testPipeline(pipeline, description)

hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=100000)
lr = LogisticRegression(regParam=0.1, maxIter=20)
pipeline = Pipeline(stages=[hashingTF, lr])
description = "Hashing Term Frequencies con 100000 features in output\nLogistic Regression con regParam=0.1 e maxIter=20"
testPipeline(pipeline, description)

hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=100000)
lr = LogisticRegression(regParam=0.01, maxIter=20)
pipeline = Pipeline(stages=[hashingTF, lr])
description = "Hashing Term Frequencies con 100000 features in output\nLogistic Regression con regParam=0.01 e maxIter=20"
testPipeline(pipeline, description)

hashingTF = HashingTF(inputCol="words", outputCol="rawfeatures", numFeatures=100000)
idf = IDF(inputCol="rawfeatures", outputCol="features")
lr = LogisticRegression(regParam=0.1, maxIter=20)
pipeline = Pipeline(stages=[hashingTF, idf, lr])
description = "Hashing Term Frequencies con 100000 features in output\nIDF con minDocFreq=0\nLogistic Regression con regParam=0.1 e maxIter=20"
testPipeline(pipeline, description)

hashingTF = HashingTF(inputCol="words", outputCol="rawfeatures", numFeatures=100000)
idf = IDF(inputCol="rawfeatures", outputCol="features", minDocFreq=50)
lr = LogisticRegression(regParam=0.1, maxIter=20)
pipeline = Pipeline(stages=[hashingTF, idf, lr])
description = "Hashing Term Frequencies con 100000 features in output\nIDF con minDocFreq=50\nLogistic Regression con regParam=0.1 e maxIter=20"
testPipeline(pipeline, description)

hashingTF = HashingTF(inputCol="words", outputCol="rawfeatures", numFeatures=1000)
idf = IDF(inputCol="rawfeatures", outputCol="features", minDocFreq=0)
lr = LogisticRegression(regParam=0.1, maxIter=20)
pipeline = Pipeline(stages=[hashingTF, idf, lr])
description = "Hashing Term Frequencies con 1000 features in output\nIDF con minDocFreq=0\nLogistic Regression con regParam=0.1 e maxIter=20"
testPipeline(pipeline, description)

try:
    filterer = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawfeatures", numFeatures=1000)
    idf = IDF(inputCol="rawfeatures", outputCol="features", minDocFreq=0)
    lr = LogisticRegression(regParam=0.1, maxIter=20)
    pipeline = Pipeline(stages=[filterer, hashingTF, idf, lr])
    description = "StopWordRemover\nHashing Term Frequencies con 1000 features in output\nIDF con minDocFreq=0\nLogistic Regression con regParam=0.1 e maxIter=20"
    testPipeline(pipeline, description)
except:
    pass

hashingTF = HashingTF(inputCol="words", outputCol="rawfeatures", numFeatures=1000)
idf = IDF(inputCol="rawfeatures", outputCol="features", minDocFreq=50)
lr = LogisticRegression(regParam=0.1, maxIter=20)
pipeline = Pipeline(stages=[hashingTF, idf, lr])
description = "Hashing Term Frequencies con 1000 features in output\nIDF con minDocFreq=50\nLogistic Regression con regParam=0.1 e maxIter=20"
testPipeline(pipeline, description)

#RANDOM FOREST
hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=1000)
lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
pipeline = Pipeline(stages=[hashingTF, lr])
description = "Hashing Term Frequencies con 1000 features in output\nRandom Forest con maxDepth=5 e numTrees=20"
testPipeline(pipeline, description)

#hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=100000)
#lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
#pipeline = Pipeline(stages=[hashingTF, lr])
#description = "Hashing Term Frequencies con 100000 features in output\nRandom Forest con maxDepth=5 e numTrees=20"
#testPipeline(pipeline, description)


spark.stop()
spark = getSpark()
hashingTF = HashingTF(inputCol="words", outputCol="rawfeatures", numFeatures=1000)
idf = IDF(inputCol="rawfeatures", outputCol="features", minDocFreq=0)
lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
pipeline = Pipeline(stages=[hashingTF, idf, lr])
description = "Hashing Term Frequencies con 1000 features in output\nIDF con minDocFreq=0\nRandom Forest con maxDepth=5 e numTrees=20"
testPipeline(pipeline, description)

spark.stop()
spark = getSpark()
hashingTF = HashingTF(inputCol="words", outputCol="rawfeatures", numFeatures=1000)
idf = IDF(inputCol="rawfeatures", outputCol="features", minDocFreq=50)
lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
pipeline = Pipeline(stages=[hashingTF, idf, lr])
description = "Hashing Term Frequencies con 1000 features in output\nIDF con minDocFreq=50\nRandom Forest con maxDepth=5 e numTrees=20"
testPipeline(pipeline, description)

spark.stop()
spark = getSpark()
hashingTF = HashingTF(inputCol="words", outputCol="rawfeatures", numFeatures=1000)
idf = IDF(inputCol="rawfeatures", outputCol="features", minDocFreq=50)
lr = RandomForestClassifier(maxDepth=8, numTrees=40, seed=42)
pipeline = Pipeline(stages=[hashingTF, idf, lr])
description = "Hashing Term Frequencies con 1000 features in output\nIDF con minDocFreq=50\nRandom Forest con maxDepth=8 e numTrees=40"
testPipeline(pipeline, description)

try:
    spark.stop()
    spark = getSpark()
    filterer = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawfeatures", numFeatures=1000)
    idf = IDF(inputCol="rawfeatures", outputCol="features", minDocFreq=0)
    lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
    pipeline = Pipeline(stages=[filterer, hashingTF, idf, lr])
    description = "StopWordRemover\nHashing Term Frequencies con 1000 features in output\nIDF con minDocFreq=0\nRandom Forest con regParam=0.1 e maxIter=20"
    testPipeline(pipeline, description)
except:
    pass


try:
    spark.stop()
    spark = getSpark()
    w2v = Word2Vec(inputCol="words", outputCol="features") 
    lr = LogisticRegression(regParam=0.1, maxIter=20, seed=42)
    pipeline = Pipeline(stages=[w2v, lr])
    description = "Word2Vec dal nostro dataset\nLogistic Regression con regParam=0.1 e maxIter=20"
    testPipeline(pipeline, description)
except:
    pass

try:
    spark.stop()
    spark = getSpark()
    w2v = Word2Vec(inputCol="words", outputCol="w2v") 
    filterer = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawfeatures", numFeatures=1000)
    idf = IDF(inputCol="rawfeatures", outputCol="idfs", minDocFreq=0)
    va = VectorAssembler(inputCols=["w2v", "idfs"], outputCol="features") 
    lr = LogisticRegression(regParam=0.1, maxIter=20, seed=42)
    pipeline = Pipeline(stages=[w2v, filterer, hashingTF, idf, va, lr])
    description = "Word2Vec dal nostro dataset\nStopWord,hashingTF con 1000, idf con 0\nVectorAssembler per unire w2c e tfidf\nLogistic Regression con regParam=0.1 e maxIter=20"
    testPipeline(pipeline, description)
except:
    pass

try:
    spark.stop()
    spark = getSpark()
    w2v = Word2Vec(inputCol="words", outputCol="features") 
    lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
    pipeline = Pipeline(stages=[w2v, lr])
    description = "Word2Vec dal nostro dataset\nRandom Forest con maxDepth=5 e numTrees=20"
    testPipeline(pipeline, description)
except:
    pass

try:
    spark.stop()
    spark = getSpark()
    w2v = Word2Vec(inputCol="words", outputCol="w2v") 
    filterer = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawfeatures", numFeatures=1000)
    idf = IDF(inputCol="rawfeatures", outputCol="idfs", minDocFreq=0)
    va = VectorAssembler(inputCols=["w2v", "idfs"], outputCol="features") 
    lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
    pipeline = Pipeline(stages=[w2v, filterer, hashingTF, idf, va, lr])
    description = "Word2Vec dal nostro dataset\nStopWord,hashingTF con 1000, idf con 0\nVectorAssembler per unire w2c e tfidf\nRandom Forest con maxDepth=5 e numTrees=20"
    testPipeline(pipeline, description)
except:
    pass

spark.stop()
spark = getSpark()
hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=100000)
lr = RandomForestClassifier(maxDepth=8, numTrees=40, seed=42)
pipeline = Pipeline(stages=[hashingTF, lr])
description = "Hashing Term Frequencies con 100000 features in output\nRandom Forest con maxDepth=8 e numTrees=40"
testPipeline(pipeline, description)

spark.stop()
spark = getSpark()
hashingTF = HashingTF(inputCol="words", outputCol="rawfeatures", numFeatures=100000)
idf = IDF(inputCol="rawfeatures", outputCol="features")
lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
pipeline = Pipeline(stages=[hashingTF, idf, lr])
description = "Hashing Term Frequencies con 100000 features in output\nIDF con minDocFreq=0\nRandom Forest con maxDepth=5 e numTrees=20"
testPipeline(pipeline, description)

spark.stop()
spark = getSpark()
hashingTF = HashingTF(inputCol="words", outputCol="rawfeatures", numFeatures=100000)
idf = IDF(inputCol="rawfeatures", outputCol="features", minDocFreq=50)
lr = RandomForestClassifier(maxDepth=8, numTrees=40, seed=42)
pipeline = Pipeline(stages=[hashingTF, idf, lr])
description = "Hashing Term Frequencies con 100000 features in output\nIDF con minDocFreq=50\nRandom Forest con maxDepth=8 e numTrees=40"
testPipeline(pipeline, description)