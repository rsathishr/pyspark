from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, year, round, when, rank
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType

# Initialize Spark Session
spark = SparkSession.builder.appName("Question3_Assessment").getOrCreate()

# 1. Define Schemas (since files do not have headers per Image 2)
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

customers_schema = StructType([
    StructField("customer_id", IntegerType()),
    StructField("first_name", StringType()),
    StructField("last_name", StringType()),
    StructField("email", StringType()),
    StructField("phone", StringType()),
    StructField("address", StringType()),
    StructField("city", StringType()),
    StructField("state", StringType()),
    StructField("zip", StringType())
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

departments_schema = StructType([
    StructField("department_id", IntegerType()),
    StructField("department", StringType())
])

# 2. Load Datasets
orders = spark.read.schema(orders_schema).csv("orders.csv")
order_items = spark.read.schema(order_items_schema).csv("order_items.csv")
customers = spark.read.schema(customers_schema).csv("customers.csv")
products = spark.read.schema(products_schema).csv("products.csv")
categories = spark.read.schema(categories_schema).csv("categories.csv")
departments = spark.read.schema(departments_schema).csv("departments.csv")

# 3. Join Datasets and Filter
# Join path: order_items -> orders -> customers AND order_items -> products -> categories -> departments
joined_df = order_items.join(orders, "order_id") \
    .join(customers, "customer_id") \
    .join(products, "product_id") \
    .join(categories, "category_id") \
    .join(departments, "department_id")

# Filter for status 'COMPLETE' and target departments
filtered_df = joined_df.filter(
    (col("order_status") == "COMPLETE") & 
    (col("department").isin("Fitness", "Golf"))
)

# 4. Extract Year and Aggregate Sales
# total_sales is computed as sum of total_price
sales_by_year = filtered_df.withColumn("order_year", year(col("order_date"))) \
    .filter(col("order_year").isin(2013, 2014)) \
    .groupBy("department", "state", "order_year") \
    .agg(sum("total_price").alias("yearly_sales"))

# 5. Pivot to get 2013 and 2014 sales side-by-side
pivoted_df = sales_by_year.groupBy("department", "state") \
    .pivot("order_year", [2013, 2014]) \
    .sum("yearly_sales") \
    .withColumnRenamed("2013", "2013_total_sales") \
    .withColumnRenamed("2014", "2014_total_sales")

# 6. Discard data if no sales in either 2013 or 2014
clean_df = pivoted_df.dropna(subset=["2013_total_sales", "2014_total_sales"])

# 7. Calculate Growth% and Rounding
# Formula: ((2014 - 2013) / 2013) * 100
final_calc_df = clean_df.withColumn(
    "growth%", 
    round(((col("2014_total_sales") - col("2013_total_sales")) / col("2013_total_sales")) * 100, 2)
).withColumn("2013_total_sales", round(col("2013_total_sales"), 1)) \
 .withColumn("2014_total_sales", round(col("2014_total_sales"), 1))

# 8. Fetch top 3 states for each department based on growth%
window_spec = Window.partitionBy("department").orderBy(col("growth%").desc())

top_3_df = final_calc_df.withColumn("rank", rank().over(window_spec)) \
    .filter(col("rank") <= 3) \
    .select("department", "state", "2014_total_sales", "2013_total_sales", "growth%")

# 9. Store the output in "question3" directory as CSV
# Coalesce to 1 to ensure a single file output if needed for small datasets
top_3_df.coalesce(1).write.mode("overwrite").option("header", "true").csv("question3")

# Show output for verification
top_3_df.show()
