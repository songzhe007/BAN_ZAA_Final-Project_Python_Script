import bq_helper
import csv

from google.cloud import bigquery

from bq_helper import BigQueryHelper

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")


open_aq.list_tables()



client = bigquery.Client()

query = """ SELECT EXTRACT(YEAR FROM timestamp) as `Year`,
                   AVG(value) as `Average`,
                   country,
                   source_name,
                   city,
                   unit,
                   location,
                   latitude,
                   longitude,
                   timestamp
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'CA' OR country = 'US'
        GROUP BY Year, 
                 latitude,
                 longitude,
                 country,
                 source_name,
                 unit,
                 location,
                 timestamp,
                 city
        LIMIT 2000000
        """

query_job = client.query(query) 

print("The query data:")

table_id='bigquery-public-data.openaq.global_air_quality'




table = client.get_table(table_id)

# View table properties
print(
    "Got table '{}.{}.{}'.".format(table.project, table.dataset_id, table.table_id)
)
print("Table schema: {}".format(table.schema))
print("Table description: {}".format(table.description))
print("Table has {} rows".format(table.num_rows))


csv_file_name = "dataset_aq.csv"


with open(csv_file_name, mode="w", newline="") as csv_file:

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(["Year", "Value", "City", "Province", "Country", "Unit"])

    for row in query_job:
        csv_writer.writerow([row["Year"], row["Average"], row["location"], row["city"], row["country"], row["unit"], row["timestamp"]])

        
        
        
print(f"write completed! {csv_file_name}")