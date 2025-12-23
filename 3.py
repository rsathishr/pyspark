from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, sum as _sum, round as _round, desc
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# Start Spark session
spark = SparkSession.builder.getOrCreate()

# Load CSVs (assuming no headers)
orders = spark.read.csv("orders.csv", header=False, inferSchema=True).toDF("order_id", "order_date", "customer_id", "order_status")
order_items = spark.read.csv("order_items.csv", header=False, inferSchema=True).toDF("order_item_id", "order_id", "product_id", "quantity", "total_price", "unit_price")
products = spark.read.csv("products.csv", header=False, inferSchema=True).toDF("product_id", "category_id", "product", "type", "unit_price", "url")
categories = spark.read.csv("categories.csv", header=False, inferSchema=True).toDF("category_id", "department_id", "category")
departments = spark.read.csv("departments.csv", header=False, inferSchema=True).toDF("department_id", "department")
customers = spark.read.csv("customers.csv", header=False, inferSchema=True).toDF("customer_id", "first_name", "last_name", "email", "phone", "address", "city", "state", "zip")

# Filter orders with status COMPLETE
orders_filtered = orders.filter(col("order_status") == "COMPLETE")

# Join all datasets
joined = orders_filtered.join(order_items, "order_id") \
    .join(products, "product_id") \
    .join(categories, "category_id") \
    .join(departments, "department_id") \
    .join(customers, "customer_id")

# Filter departments
filtered = joined.filter(col("department").isin("Fitness", "Golf"))

# Add year column
filtered = filtered.withColumn("year", year("order_date"))

# Aggregate total sales by department, state, year
agg_df = filtered.groupBy("department", "state", "year").agg(_sum("total_price").alias("year_sales"))

# Pivot to get 2013 and 2014 sales
pivot_df = agg_df.groupBy("department", "state") \
    .pivot("year", [2013, 2014]) \
    .agg(_sum("year_sales")) \
    .withColumnRenamed("2013", "2013_total_sales") \
    .withColumnRenamed("2014", "2014_total_sales")

# Filter valid rows with sales in both years
valid_df = pivot_df.filter((col("2013_total_sales").isNotNull()) & (col("2014_total_sales").isNotNull()) & (col("2013_total_sales") > 0))

# Calculate growth%
result_df = valid_df.withColumn("growth%", _round((col("2014_total_sales") - col("2013_total_sales")) / col("2013_total_sales") * 100, 2)) \
    .withColumn("2013_total_sales", _round(col("2013_total_sales"), 1)) \
    .withColumn("2014_total_sales", _round(col("2014_total_sales"), 1))

# Rank top 3 states per department
window_spec = Window.partitionBy("department").orderBy(desc("growth%"))
ranked_df = result_df.withColumn("rank", row_number().over(window_spec)).filter(col("rank") <= 3).drop("rank")

# Select and sort final columns
final_df = ranked_df.select("department", "state", "2014_total_sales", "2013_total_sales", "growth%") \
    .orderBy("department", desc("growth%"))

# Write output
final_df.write.csv("question3", mode="overwrite", header=True)
