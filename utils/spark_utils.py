import pandas as pd
import os

# We use PySpark for heavy processing but fall back gracefully
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


def get_spark():
    if not SPARK_AVAILABLE:
        return None
    try:
        spark = SparkSession.builder \
            .appName("HiringBiasAnalyzer") \
            .config("spark.ui.showConsoleProgress", "false") \
            .config("spark.driver.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        return spark
    except Exception:
        return None


def spark_process_jobs(df: pd.DataFrame, spark=None):
    """Process job postings with Spark, returns pandas df."""
    if spark is None:
        return _pandas_process_jobs(df)
    try:
        sdf = spark.createDataFrame(df.fillna("Unknown"))
        # Explode skills into rows
        sdf = sdf.withColumn("skill", F.explode(F.split(F.col("skills_required"), ",")))
        sdf = sdf.withColumn("skill", F.trim(F.col("skill")))
        sdf = sdf.filter(F.col("skill") != "")
        result = sdf.groupBy("industry", "skill", "year") \
                    .agg(F.count("*").alias("demand_count")) \
                    .orderBy(F.desc("demand_count"))
        return result.toPandas()
    except Exception:
        return _pandas_process_jobs(df)


def spark_process_candidates(df: pd.DataFrame, spark=None):
    """Process candidate data with Spark, returns pandas df."""
    if spark is None:
        return _pandas_process_candidates(df)
    try:
        sdf = spark.createDataFrame(df.fillna({"skills": "", "industry": "Unknown",
                                                "applied_role": "Unknown", "hired": 0}))
        result = sdf.groupBy("industry", "applied_role", "hired") \
                    .agg(F.count("*").alias("count")) \
                    .orderBy(F.desc("count"))
        return result.toPandas()
    except Exception:
        return _pandas_process_candidates(df)


def _pandas_process_jobs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        skills_raw = str(row.get("skills_required", ""))
        for skill in [s.strip() for s in skills_raw.split(",") if s.strip()]:
            rows.append({
                "industry": row.get("industry", "Unknown"),
                "skill": skill,
                "year": row.get("year", 2023),
                "demand_count": 1
            })
    if not rows:
        return pd.DataFrame(columns=["industry", "skill", "year", "demand_count"])
    result = pd.DataFrame(rows)
    return result.groupby(["industry", "skill", "year"]).sum().reset_index()


def _pandas_process_candidates(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["industry", "applied_role", "hired"]) \
             .size().reset_index(name="count")
