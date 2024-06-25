import pandas as pd
import geopandas as gpd
from shapely import wkt
from pymssql import connect
from sqlalchemy import create_engine, text
import urllib
import pyodbc
import toml


def load_elmer_table(table_name, sql=None):
    conn_string = "DRIVER={ODBC Driver 17 for SQL Server}; SERVER=AWS-PROD-SQL\Sockeye; DATABASE=Elmer; trusted_connection=yes"
    sql_conn = pyodbc.connect(conn_string)
    params = urllib.parse.quote_plus(conn_string)
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    if sql is None:
        sql = "SELECT * FROM " + table_name

    # df = pd.DataFrame(engine.connect().execute(text(sql)))
    with engine.begin() as connection:
        result = connection.execute(sql)
        df = pd.DataFrame(result.fetchall())
        df.columns = result.keys()

    return df


def load_elmer_geo(table_name):
    """Load ElmerGeo feature class as geodataframe."""

    con = connect("AWS-Prod-SQL\Sockeye", database="ElmerGeo")

    cursor = con.cursor()
    feature_class_name = table_name
    geo_col_stmt = (
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME="
        + "'"
        + feature_class_name
        + "'"
        + " AND DATA_TYPE='geometry'"
    )
    geo_col = str(pd.read_sql(geo_col_stmt, con).iloc[0, 0])
    query_string = (
        "SELECT *,"
        + geo_col
        + ".STGeometryN(1).ToString()"
        + " FROM "
        + feature_class_name
    )
    df = pd.read_sql(query_string, con)

    df.rename(columns={"": "geometry"}, inplace=True)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry")

    con.close()

    return gdf
