from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, year, round, when, desc
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType

# 1. Initialize Spark Session
spark = SparkSession.builder.appName("Question4").getOrCreate()

# 2. Define Schemas (Strictly no headers as per assessment instructions)
orders_schema = StructType([
    StructField("order_id", IntegerType()),
    StructField("order_date", TimestampType()), # Using Timestamp to handle standard date formats
    StructField("customer_id", IntegerType()),
    StructField("order_status", StringType())
])

order_items_schema = StructType([
    StructField("order_item_id", IntegerType()),
    StructField("order_id", IntegerType()),
    StructField("product_id", IntegerType()),
    StructField("quantity", IntegerType()),
    StructField("total_price", DoubleType()),
    StructField("unit_price", DoubleType())
])

products_schema = StructType([
    StructField("product_id", IntegerType()),
    StructField("category_id", IntegerType()),
    StructField("product", StringType()),
    StructField("type", StringType()),
    StructField("unit_price", DoubleType()),
    StructField("url", StringType())
])

categories_schema = StructType([
    StructField("category_id", IntegerType()),
    StructField("department_id", IntegerType()),
    StructField("category", StringType())
])

# 3. Load Datasets from the environment
# Ensure these filenames match the files in your Jupyter environment exactly
orders = spark.read.schema(orders_schema).csv("orders.csv")
order_items = spark.read.schema(order_items_schema).csv("order_items.csv")
products = spark.read.schema(products_schema).csv("products.csv")
categories = spark.read.schema(categories_schema).csv("categories.csv")

# 4. Filter for the year 2014
orders_2014 = orders.filter(year(col("order_date")) == 2014)

# 5. Join Datasets as per ER Diagram
# categories -> products -> order_items -> orders
df_joined = categories.join(products, "category_id") \
    .join(order_items, "product_id") \
    .join(orders_2014, "order_id")

# 6. Perform Aggregation
# Incomplete orders = status NOT EQUAL to 'COMPLETE'
result_agg = df_joined.groupBy("category").agg(
    count("*").alias("total_orders"),
    sum(when(col("order_status") != "COMPLETE", 1).otherwise(0)).alias("incomplete_orders")
)

# 7. Calculate Percentage, Round, and Sort
# percentage = (incomplete / total) * 100
final_output = result_agg.withColumn(
    "percentage_incomplete_orders",
    round((col("incomplete_orders") / col("total_orders")) * 100, 1)
).sort(desc("percentage_incomplete_orders"))

# 8. Select Columns in the exact order required
final_output = final_output.select(
    "category", 
    "total_orders", 
    "incomplete_orders", 
    "percentage_incomplete_orders"
)

# 9. Save to "question4" directory as CSV with header
# Using coalesce(1) to ensure a single output file is generated in the directory
final_output.coalesce(1).write.mode("overwrite").option("header", "true").csv("question4")

# Verification: The output should have 33 rows
final_output.show(33, truncate=False)
