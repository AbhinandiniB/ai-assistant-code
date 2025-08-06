import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from main import initialize_spark_session, read_excel_file, join_metadata_and_converter, process_plate_file, normalize_raw_data

def test_initialize_spark_session():
    spark = initialize_spark_session()
    assert isinstance(spark, SparkSession), "Expected a SparkSession instance"

def test_read_excel_file_valid_file():
    spark = initialize_spark_session()
    df = read_excel_file(spark, "./data_store/HTS/Metadata.xlsx")
    assert df is not None, "Expected a non-null dataframe"
    assert "Well" in df.columns, "Expected 'Well' column in the dataframe"

def test_read_excel_file_invalid_file():
    spark = initialize_spark_session()
    with pytest.raises(Exception):
        read_excel_file(spark, "./data_store/HTS/InvalidFile.xlsx")

def test_join_metadata_and_converter_valid_data():
    spark = initialize_spark_session()
    metadata_df = spark.createDataFrame([
        {"Well": "A1", "Data": 10},
        {"Well": "A2", "Data": 20}
    ])
    converter_df = spark.createDataFrame([
        {"Well Meta": "A1", "MetaData": "X"},
        {"Well Meta": "A3", "MetaData": "Y"}
    ])
    joined_df = join_metadata_and_converter(metadata_df, converter_df)
    assert joined_df.count() == 2, "Expected 2 rows in the joined dataframe"
    assert "MetaData" in joined_df.columns, "Expected 'MetaData' column in the joined dataframe"

def test_process_plate_file():
    spark = initialize_spark_session()
    metadata_converter_df = spark.createDataFrame([
        {"Well Positions": "A1", "Plate ID": "Plate 1", "Location": "L1"}
    ])
    plate_df = spark.createDataFrame([
        {"Well Positions": "A1", "Raw data": 100, "Compound ID": "Pos C"}
    ])
    plate_df.write.format("com.crealytics.spark.excel").save("./data_store/HTS/Plate 1.xlsx")
    result_df = process_plate_file(spark, "./data_store/HTS/Plate 1.xlsx", metadata_converter_df)
    assert result_df.count() == 1, "Expected 1 row after processing plate file"
    assert "Raw data" in result_df.columns, "Expected 'Raw data' column in the result dataframe"

def test_normalize_raw_data():
    spark = initialize_spark_session()
    df = spark.createDataFrame([
        {"Plate ID": "Plate 1", "Raw data": 100, "Compound ID": "Pos C"},
        {"Plate ID": "Plate 1", "Raw data": 50, "Compound ID": "Test"}
    ])
    normalized_df = normalize_raw_data(df)
    assert "Raw data Norm" in normalized_df.columns, "Expected 'Raw data Norm' column in the normalized dataframe"
    normalized_value = normalized_df.filter(col("Compound ID") == "Test").select("Raw data Norm").collect()[0][0]
    assert normalized_value == 50.0, "Expected normalized value to be 50.0"
 