






from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

sc.install_pypi_package('sklearn')


from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
data = fetch_covtype()
X = data["data"]
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('--hello-')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_t = scaler.fit_transform(X_train)
X_test_t = scaler.transform(X_test)
print('-hello-')

sc.install_pypi_package('pandas')
import pandas as pd
pandas_df = pd.DataFrame(X_train_t)
pandas_df["label"] = y_train
spark_df = spark.createDataFrame(pandas_df)
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(
    inputCols=[str(a) for a in pandas_df.columns[:-1]],
    outputCol="features"
)
spark_DataF = assembler.transform(spark_df)
print('-hello--')
import time
start = time.time()
from pyspark.ml.classification import LogisticRegression as LR
lr = LR()
cvModel = lr.fit(spark_DataF)
print("-- spark ML LR --")
print("Train Time: {0}".format(time.time() - start))
# print("Best Model CV Score: {0}".format(np.mean(cvModel.avgMetrics)))

# test holdout
pandas_df = pd.DataFrame(X_test_t)
pandas_df["label"] = y_test
eval_df = spark.createDataFrame(pandas_df)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print("Holdout F1: {0}".format(evaluator.evaluate(cvModel.transform(spark_DataF))))
