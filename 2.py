from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum, year, round
)

spark = SparkSession.builder.getOrCreate()

# -----------------------------
# 1. Read CSV files (NO HEADERS)
# -----------------------------

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

# -----------------------------------------
# 2. Filter COMPLETE orders for 2013 & 2014
# -----------------------------------------

orders_filtered = orders \
    .filter(col("order_status") == "COMPLETE") \
    .withColumn("year", year(col("order_date"))) \
    .filter(col("year").isin(2013, 2014))

# -----------------------------------------
# 3. Join all tables
# -----------------------------------------

joined_df = orders_filtered \
    .join(order_items, "order_id") \
    .join(products, "product_id") \
    .join(categories, "category_id") \
    .join(departments, "department_id") \
    .filter(col("department").isin("Sports", "Apparel", "Fitness"))

qty_df = joined_df.groupBy(
    "department", "product", "year"
).agg(
    sum("quantity").alias("total_qty")
)

pivot_df = qty_df.groupBy(
    "department", "product"
).pivot("year", [2013, 2014]) \
 .sum("total_qty") \
 .na.fill(0)

# Rename columns
pivot_df = pivot_df \
    .withColumnRenamed("2013", "2013_qty") \
    .withColumnRenamed("2014", "2014_qty")


growth_df = pivot_df \
    .filter(col("2013_qty") > 0) \
    .withColumn(
        "growth%",
        round(((col("2014_qty") - col("2013_qty")) / col("2013_qty")) * 100)
    ) \
    .filter(col("growth%") > 50)


final_df = growth_df.select(
    "department",
    "product",
    "2013_qty",
    "2014_qty",
    "growth%"
).orderBy(
    col("department").asc(),
    col("growth%").desc()
)


final_df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("question2")
