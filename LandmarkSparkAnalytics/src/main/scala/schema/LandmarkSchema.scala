package schema

import org.apache.spark.sql.types._

object LandmarkSchema {
  val schema: StructType = StructType(Array(
    StructField("landmark_id", StringType, nullable = true),
      StructField("category", StringType, nullable = true),
      StructField("supercategory", StringType, nullable = true),
      StructField("hierarchical_label", StringType, nullable = true),
      StructField("natural_or_human_made", StringType, nullable = true)
  ))
}
