package store

import org.apache.spark.sql.SparkSession
import schema.LandmarkSchema

object LandmarkDataFrame {
  val spark = SparkSession.builder()
    .appName("Landmark Analytics")
    .master("local[*]") // Use local mode with all cores
    .getOrCreate()

  // Read data using the defined schema
//  var df = spark.read
//    .option("header", "true") // Assumes the first row is a header
//    .schema(LandmarkSchema.schema)
//    .csv("./train_label_to_hierarchical.csv")

}
