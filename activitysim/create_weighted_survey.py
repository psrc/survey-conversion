import os
import pandas as pd
import pyodbc
import urllib
import sqlalchemy

# input_dir = r'R:\e2projects_two\activitysim\override\formatted_data\tues_wed_thur'
input_dir = r'C:\Workspace\psrc_activitysim_master\data\survey_data'

hh_wt_col = 'hh_weight_2017_2019'
person_wt_col = 'hh_weight_2017_2019'   # no specific person weight available, use HH weight
trip_wt_col = 'trip_weight_2017_2019'

# Database connections for PSRC Elmer DB
conn_string = "DRIVER={ODBC Driver 17 for SQL Server}; SERVER=AWS-PROD-SQL\Sockeye; DATABASE=Elmer; trusted_connection=yes"
sql_conn = pyodbc.connect(conn_string)
params = urllib.parse.quote_plus(conn_string)
engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

# Load mapping between original household and person IDs
mapping_df = pd.read_csv(os.path.join(input_dir, 'person_and_household_id_mapping.csv'))

##############################
# Household
##############################

# Add household weights
hh = pd.read_csv(os.path.join(input_dir, 'override_households.csv'))
hh = hh.merge(mapping_df, left_on='household_id', right_on='household_id')

# Households are duplicated from different days
# Take households only from the first day
hh['household_id_elmer'] = hh['household_id_original'].apply(lambda x: x.split('_')[0]).astype('int64')
hh['survey_day'] = hh['household_id_original'].apply(lambda x: x.split('_')[1])

hh = hh[hh['survey_day'] == hh['survey_day'].min()]

# Join weights from Elmer
hh_elmer = pd.read_sql(sql='SELECT * FROM HHSurvey.v_households WHERE survey_year IN (2017, 2019)', con=engine)

hh = hh.merge(hh_elmer[['household_id', hh_wt_col]], left_on='household_id_elmer', right_on='household_id')


##############################
# Person
##############################

# Add person weights
person = pd.read_csv(os.path.join(input_dir, 'override_persons.csv'))
person = person.merge(mapping_df, on='person_id')

person['person_id_elmer'] = person['person_id_original'].apply(lambda x: x.split('_')[0]).astype('int64')
person['survey_day'] = person['person_id_original'].apply(lambda x: x.split('_')[1])

person = person[person['survey_day'] == person['survey_day'].min()]

# Join weights from Elmer
person_elmer = pd.read_sql(sql='SELECT * FROM HHSurvey.v_persons WHERE survey_year IN (2017, 2019)', con=engine)

person = person.merge(person_elmer[['person_id', person_wt_col]], left_on='person_id_elmer', 
         right_on='person_id')

##############################
# Trip
##############################

# Add trip weights
trip = pd.read_csv(os.path.join(input_dir, 'override_trips.csv'))
trip_elmer = pd.read_sql(sql='SELECT * FROM HHSurvey.v_trips WHERE survey_year IN (2017, 2019)', con=engine)

trip_mapping_df = pd.read_csv(os.path.join(input_dir, 'trip_id_mapping.csv'))
trip = trip.merge(trip_mapping_df[['trip_id', 'trip_id_elmer']], on='trip_id')

trip = trip.merge(trip_elmer[['trip_id', trip_wt_col]], left_on='trip_id_elmer', right_on='trip_id')

##############################
# Tour
##############################
tour = pd.read_csv(os.path.join(input_dir, 'override_tours.csv'))

tour_mean_df = trip.groupby('tour_id').mean()[trip_wt_col]
tour_mean_df = tour_mean_df.reset_index()