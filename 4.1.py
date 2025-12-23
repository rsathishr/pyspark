from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, countDistinct, round
from pyspark.sql.types import FloatType

spark = SparkSession.builder.appName("Question4").getOrCreate()

# ---------------- Load CSVs (NO HEADERS) ----------------
orders = spark.read.csv("orders.csv", header=False, inferSchema=True) \
    .toDF("order_id","order_date","customer_id","order_status")

order_items = spark.read.csv("order_items.csv", header=False, inferSchema=True) \
    .toDF("order_item_id","order_id","product_id","quantity","total_price","unit_price")

products = spark.read.csv("products.csv", header=False, inferSchema=True) \
    .toDF("product_id","category_id","product","type","unit_price")

categories = spark.read.csv("categories.csv", header=False, inferSchema=True) \
    .toDF("category_id","department_id","category")

# ---------------- Join as per ER diagram ----------------
df = orders \
    .join(order_items, "order_id") \
    .join(products, "product_id") \
    .join(categories, "category_id")

# ---------------- Filter only 2014 ----------------
df_2014 = df.withColumn("year", year(col("order_date"))) \
            .filter(col("year") == 2014)

# ---------------- Aggregate ----------------
agg_df = df_2014.groupBy("category").agg(
    countDistinct("order_id").alias("total_orders"),
    countDistinct(
        col("order_id")
    ).alias("dummy")  # placeholder
)

# ---------------- Count incomplete orders ----------------
incomplete_df = df_2014.filter(col("order_status") != "COMPLETE") \
    .groupBy("category") \
    .agg(countDistinct("order_id").alias("incomplete_orders"))

# ---------------- Join counts ----------------
final_df = agg_df.join(incomplete_df, "category", "left") \
    .fillna(0, ["incomplete_orders"])

# ---------------- Calculate percentage ----------------
final_df = final_df.withColumn(
    "percentage_incomplete_orders",
    round(
        (col("incomplete_orders") / col("total_orders")) * 100, 1
    ).cast(FloatType())
)

# ---------------- Final Select & Sort ----------------
result_df = final_df.select(
    "category",
    "total_orders",
    "incomplete_orders",
    "percentage_incomplete_orders"
).orderBy(col("percentage_incomplete_orders").desc())

# ---------------- Save Output ----------------
result_df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("question4")
