from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum, year, round,
    row_number
)
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()


orders = spark.read.csv("orders.csv", inferSchema=True) \
    .toDF("order_id", "order_date", "customer_id", "order_status")

order_items = spark.read.csv("order_items.csv", inferSchema=True) \
    .toDF("order_item_id", "order_id", "product_id", "quantity", "total_price", "unit_price")

products = spark.read.csv("products.csv", inferSchema=True) \
    .toDF("product_id", "category_id", "product", "type", "unit_price", "url")

categories = spark.read.csv("categories.csv", inferSchema=True) \
    .toDF("category_id", "department_id", "category")

departments = spark.read.csv("departments.csv", inferSchema=True) \
    .toDF("department_id", "department")


orders_filtered = orders \
    .filter(col("order_status") == "COMPLETE") \
    .withColumn("year", year(col("order_date"))) \
    .filter(col("year") == 2013)


joined_df = orders_filtered \
    .join(order_items, "order_id") \
    .join(products, "product_id") \
    .join(categories, "category_id") \
    .join(departments, "department_id") \
    .filter(col("department").isin("Apparel", "Fitness"))


agg_df = joined_df.groupBy(
    "year", "department", "category", "product"
).agg(
    sum("quantity").alias("total_quantity"),
    round(sum("total_price")).alias("total_sale_value")
)


window_spec = Window.partitionBy(
    "department", "category"
).orderBy(col("total_sale_value").desc())

final_df = agg_df \
    .withColumn("rn", row_number().over(window_spec)) \
    .filter(col("rn") == 1) \
    .drop("rn")

final_df = final_df.orderBy(
    col("department").asc(),
    col("category").asc()
)


final_df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("question1")

