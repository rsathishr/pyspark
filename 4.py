from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, year, round, when
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType

# 1. Initialize Spark Session
spark = SparkSession.builder.appName("Question4_Assessment").getOrCreate()

# 2. Define Schemas (Based on ER Diagram; files have no headers)
orders_schema = StructType([
    StructField("order_id", IntegerType()),
    StructField("order_date", DateType()),
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

# 3. Load Datasets
orders = spark.read.schema(orders_schema).csv("orders.csv")
order_items = spark.read.schema(order_items_schema).csv("order_items.csv")
products = spark.read.schema(products_schema).csv("products.csv")
categories = spark.read.schema(categories_schema).csv("categories.csv")

# 4. Filter for orders placed in the year 2014
orders_2014 = orders.filter(year(col("order_date")) == 2014)

# 5. Join datasets per the ER diagram
# Join path: categories -> products -> order_items -> orders
joined_df = categories.join(products, "category_id") \
    .join(order_items, "product_id") \
    .join(orders_2014, "order_id")

# 6. Aggregate data per category
# total_orders: Total count of line items in that category
# incomplete_orders: Items where order_status != 'COMPLETE'
agg_df = joined_df.groupBy("category").agg(
    count("*").alias("total_orders"),
    sum(when(col("order_status") != "COMPLETE", 1).otherwise(0)).alias("incomplete_orders")
)

# 7. Calculate percentage, round to one decimal place, and sort descending
final_df = agg_df.withColumn(
    "percentage_incomplete_orders", 
    round((col("incomplete_orders") / col("total_orders")) * 100, 1)
).orderBy(col("percentage_incomplete_orders").desc())

# Select final columns in correct order
result_df = final_df.select(
    "category", 
    "total_orders", 
    "incomplete_orders", 
    "percentage_incomplete_orders"
)

# 8. Store output in "question4" directory as CSV with header
# We use coalesce(1) to ensure the output matches the 33-row requirement in a single file
result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv("question4")

# Show output to verify it matches sample (limit to 33 rows)
result_df.show(33, truncate=False)
