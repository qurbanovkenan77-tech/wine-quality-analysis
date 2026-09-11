import time
import polars as pl


# load the dataset and measure load time
start = time.perf_counter()

df = pl.read_csv("wine_quality_merged.csv")

end = time.perf_counter()

print("\nPolars CSV Load Time:")
print(end - start)


# display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())


# inspect the data
print("\nDataset Schema:")
print(df.schema)

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.null_count())

print("\nNumber of Duplicates:")
print(df.is_duplicated().sum())


# basic filtering and grouping
start = time.perf_counter()

high_quality = df.filter(pl.col("quality") >= 7)

average_quality = (
    df.group_by("type")
    .agg(
        pl.col("quality")
        .mean()
        .alias("average_quality")
    )
    .sort("type")
)

end = time.perf_counter()

print("\nPolars Analysis Time:")
print(end - start)


print("\nHigh Quality Wines:")
print(high_quality.head())

print("\nNumber of High Quality Wines:")
print(high_quality.height)

print("\nAverage Quality by Wine Type:")
print(average_quality)