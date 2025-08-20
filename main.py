from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, lit, split

# Define constants to avoid literal duplication
COLUMN_PLATE_ID = "Plate ID"
COLUMN_WELL_POSITIONS = "Well Positions"
COLUMN_RAW_DATA = "Raw data"

def initialize_spark_session():
    return SparkSession.builder.appName("HighThroughputSystem").config("spark.jars.packages", "com.crealytics:spark-excel_2.12:0.15.1").getOrCreate()

def read_excel_file(spark, file_path):
    return spark.read.format("com.crealytics.spark.excel").option("header", "true").option("inferSchema", "true").load(file_path)

def join_metadata_and_converter(metadata_df, converter_df):
    return metadata_df.join(converter_df, metadata_df["Well"] == converter_df["Well Meta"], "left_outer")

def process_plate_file(spark, file_path, metadata_converter_df):
    plate_df = read_excel_file(spark, file_path)
    plate_id = file_path.split('/')[-1].split('.')[0]
    plate_df = plate_df.withColumn(COLUMN_PLATE_ID, lit(plate_id))
    joined_df = metadata_converter_df.join(plate_df,
                                           (metadata_converter_df[COLUMN_WELL_POSITIONS] == plate_df[COLUMN_WELL_POSITIONS]) &
                                           (metadata_converter_df[COLUMN_PLATE_ID] == plate_df[COLUMN_PLATE_ID]),
                                           "inner")
    return joined_df.select(COLUMN_WELL_POSITIONS, COLUMN_RAW_DATA, "Location", COLUMN_PLATE_ID, "Well Meta", "Compound ID", "Library")        

def normalize_raw_data(df):
    mean_df = df.filter(col("Compound ID") == "Pos C").groupBy(COLUMN_PLATE_ID).agg(mean(COLUMN_RAW_DATA).alias("mean_raw_data"))
    normalized_df = df.join(mean_df, COLUMN_PLATE_ID).withColumn("Raw data Norm", (col(COLUMN_RAW_DATA) / col("mean_raw_data")) * 100).drop("mean_raw_data")
    return normalized_df

def main():
    spark = initialize_spark_session()

    metadata_df = read_excel_file(spark, "./data_store/HTS/Metadata.xlsx")
    converter_df = read_excel_file(spark, "./data_store/HTS/Well converter.xlsx")
    metadata_converter_df = join_metadata_and_converter(metadata_df, converter_df)

    plate_df = process_plate_file(spark, "./data_store/HTS/Plate 1.xlsx", metadata_converter_df)

    normalized_df = normalize_raw_data(plate_df)

    normalized_df.show()

if __name__ == "__main__":
    main()