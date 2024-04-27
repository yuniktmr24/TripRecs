package query

import org.apache.spark.sql.SparkSession

object LandmarkQueryRepository {

  def rankSuperCategory(spark: SparkSession): Unit = {
    println("Rank Supercategory")

    val supercategoryCounts = spark.sql(
      """
    SELECT supercategory, COUNT(*) AS count
    FROM landmark_labels
    GROUP BY supercategory
    ORDER BY count DESC
    """)

    supercategoryCounts.show()
  }

  def rankHierarchicalLabel(spark: SparkSession): Unit = {
    println("Rank Hierarchical Label")

    val hierarchicalLabelCounts = spark.sql(
      """
      SELECT hierarchical_label, COUNT(*) AS count
      FROM landmark_labels
      GROUP BY hierarchical_label
      ORDER BY count DESC
      """)

    hierarchicalLabelCounts.show()
  }

  def rankNaturalOrHumanMade(spark: SparkSession): Unit = {
    println("Rank Natural or Human Made")

    val naturalOrHumanMadeCounts = spark.sql(
      """
      SELECT natural_or_human_made, COUNT(*) AS count
      FROM landmark_labels
      GROUP BY natural_or_human_made
      ORDER BY count DESC
      """)

    naturalOrHumanMadeCounts.show()
  }

}
