import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import datetime as dt
from itertools import chain
import altair as alt
import plotly.graph_objects as go

st.set_page_config(layout="wide")
#prefecture_codes are the prefectures and their coresponding ids
@st.cache_data
def get_prefecture_codenames():
    return{'AGRINIO': 'AGR', 'AKRINI': 'AKR', 'AKROTIRI': 'AKO', 'AMFISSA': 'AMF', 'VOLOS': 'VOII',
 'VOUTES': 'VOU', 'FINOKALIAS': 'FIN', 'IRAKLEIO': 'HER', 'IOANNINA': 'IOII', 'KAVALA': 'KAII',
 'KARPENISI': 'KAR', 'KARUOCHORI': 'KAY', 'KOZANI': 'KOZ', 'LAMIA': 'LAM', 'LARISA': 'LAR',
 'LEIBADIA': 'LEI', 'PATRA': 'PAII', 'PTOLEMAIDA': 'PTO', 'CHALKIDA': 'HAL', 'CHANIA': 'CHA','AGIAS_SOFIAS': 'AGS',
 'ARISTOTLE_UNIVERSITY_THESSALONIKI': 'APT', 'KALAMARIA': 'KAL', 'KORDELIO': 'KOD','NEOCHOROUDA': 'NEO','PANORAMA': 'PAO',
 'SINDOS': 'SIN','AGIA_PARASKEUI': 'AGP', 'ALIARTOS': 'ALI', 'ATHENS': 'ATH','ELEUSINA': 'ELE','GALATSI': 'GAL','GEOPONIKH': 'GEO',
 'GOUDI': 'GOU','KOROPI': 'KOR', 'LIOSIA': 'LIO', 'LUKOBRUSI': 'LYK', 'MAROUSI': 'MAR', 'NEA_SMURNI': 'SMY', 'OINOFYTA': 'OIN', 'PATISION': 'PAT',
 'PEIRAIAS': 'BIO', 'PERISTERI': 'PER','THRAKOMAKEDONES': 'THR', 'ZOGRAFOU': 'PAN'}
#region_names_dict is the dictionray that containts the code name for each region 
#is used to create the synergy for the prefectures each region has
@st.cache_data
def get_region_names_dict():
    return{'Eastern macedonia thrace':'EMT','West macedonia':'WEM','West greece':'WEG',
                  'Epirus':'EPR','Thessaly':'THL','Crete':'CRE','Central greece':'CEG','Attica and voiotia':'ATT',
                  'Thessaloniki':'THS'}
#region_dictionray is the one that gives us which prefecture codes each region code contains
@st.cache_data
def get_region_dict():
    return{'EMT':['KAII'],
            'WEM':['PTO','AKR','KAY','KOZ'],
            'WEG':['AGR','PAII'],
            'EPR':['IOII'],
            'THL':['VOII','LAR'],
            'CRE':['CHA','AKO','VOU','HER','FIN'],
            'CEG':['AMF','KAR','LAM','LEI','HAL'],
            #'CEM':['AGS','APT','KAL','KOD','NEO','PAO','SIN'],
            'ATT':['AGP', 'ALI', 'ATH', 'ELE', 'GAL', 'GEO', 'GOU', 'KOR', 'LIO', 'LYK',
                   'MAR', 'SMY', 'OIN', 'PAT', 'BIO', 'PER', 'THR', 'PAN','ZWG','PIR'],
            'THS':['AGS','APT','KOD','NEO','SIN','KAL','PAO']}

prefecture_codenames = get_prefecture_codenames()
region_names_dict = get_region_names_dict()
region_dict = get_region_dict()
CEM_suburbs={'THESSALONIKI CENTER':['AGIAS_SOFIAS','ARISTOTLE_UNIVERSITY_THESSALONIKI'],'WEST THESSALONIKI':['KORDELIO','NEOQWROUDA','SINDOS'],
             'EAST THESSALONIKI':['KALAMARIA'],'NORTH THESSALONIKI':['PANORAMA']}
ATT_suburbs={'ATHENIAN CENTRAL SECTOR':['ATHENS','GALATSI','ZWGRAFOU','GOUDH','PATHSIWN'],
             'ATHENIAN NORTH SECTOR':['AGIA_PARASKEUH','LUKOBRUSH','MAROUSI',' THRAKOMAKEDONES'],
             'ATHENIAN WEST SECTOR':['PERISTERI','LIOSIA'],'ATHENIAN SOUTH SECTOR':['NEA_SMURNH'],'WEST ATTICAN SECTOR':['ELEUSINA'],
             'PIREUS PREFECTURE':['PEIRAIAS']}
suburbs={'THESSALONIKI CENTER':['AGIAS_SOFIAS','ARISTOTLE_UNIVERSITY_THESSALONIKI'],'WEST THESSALONIKI':['KORDELIO','NEOQWROUDA','SINDOS'],
             'EAST THESSALONIKI':['KALAMARIA'],'NORTH THESSALONIKI':['PANORAMA'],
             'ATHENIAN CENTRAL SECTOR':['ATHENS','GALATSI','ZWGRAFOU','GOUDH','PATHSIWN'],
             'ATHENIAN NORTH SECTOR':['AGIA_PARASKEUH','LUKOBRUSH','MAROUSI',' THRAKOMAKEDONES'],
             'ATHENIAN WEST SECTOR':['PERISTERI','LIOSIA'],'ATHENIAN SOUTH SECTOR':['NEA_SMURNH'],'WEST ATTICAN SECTOR':['ELEUSINA'],
             'PIREUS PREFECTURE':['PEIRAIAS']}
@st.cache_data
def has_stepsize_one(it):
    return all(x2 - x1 == 1 for x1, x2 in zip(it[:-1], it[1:]))
@st.cache_data
def is_query_complete(prefecture_name,year_query,region):
    if  (not prefecture_name and not region in ['THC','WTH','ETH','NTH','ACS','ANS','AWS','ASS','WAS','PIP']):
        st.warning("Please select at least one prefecture.")
        return False
    if not year_query:
        st.warning("Please select at least one year.")
        return False
    return True

def column_check(col):
    if not col:
        #st.warning('Please select at least 1 column')
        return False
    return True

def get_values(dictionary, keys):
    if not isinstance(keys, (list, tuple)):  # Ensure keys are iterable
        keys = [keys]
    return list(chain.from_iterable(map(dictionary.get, keys)))

def corresponding_key(val, dictionary):
        """"This function gets the codes of the prefectures and give us the name of each prefecture code
        They will be used in the selection box where the user will have as available options for each region he choose"""
        name_list=[]
        for k, v in dictionary.items():
            if v in val:
                k=k[0]+k[1:].lower()
                k=k.replace("_"," ")
                name_list.append(k)
        if name_list:
            return name_list
        else:
            return 'No prefectures'  



class User_selection:
    
    def __init__(self, identifier, region_names_dict, region_dict,prefecture_codenames,suburbs):
        self.identifier=identifier
        self.region_names_dict = region_names_dict
        self.region_dict = region_dict
        self.prefecture_codenames = prefecture_codenames
        self.prefecture_tables=None
        self.region_id=None
        self.region_name=None
        self.prefecture_name=None
        self.dataframes={}
        self.method=None
        self.agg_dataframe=None
        self.month_query=None
        self.year_query=None
        self.timeframe=None
        self.cols=None
        self.year_list=None
        self.keystone=None
        self.suburbs=suburbs
        self.null_dataframes={}
        self.valid_column_for_suburbs=None
        
    def regions_selection(self):
        select_all = None
    # Dropdown menu for selecting a specific time_type
        if select_all:
               selected_type=st.multiselect("Please select Region/s: ",list(self.region_names_dict.keys()).sort(),
                                            default=list(self.region_names_dict.keys()).sort())
        else:
            selected_type=st.multiselect("Please select Region/s: ",sorted([key.upper() for key in self.region_names_dict.keys()]))
          # Only the selected years
        self.region_name=[regi[0]+regi[1:].lower()for regi in selected_type]
        
        if len(self.region_name)>1:
            region_ids_lst=[]
            for reg in self.region_name:
                region_ids_lst.append(self.region_names_dict[reg])
            self.region_id=region_ids_lst
            return 
        elif len(self.region_name)==1:
            self.region_id = self.region_names_dict[self.region_name[0]]
            return    
        else: 
            return 
            
        #prefecture_ids = self.region_dict.get(self.region_id, [])        
    
    def common_year_selection(self,conn):
        # Base query parts
        union_queries = []
        if self.prefecture_name:
            
            for table in self.prefecture_name:
                union_queries.append(
                    f"""SELECT DISTINCT EXTRACT(YEAR FROM "Date") AS year, '{table}' AS source_table FROM {table}"""
                )
            
            # Combine all UNION queries
            union_all_query = " UNION ALL ".join(union_queries)
        
            # Dynamic SQL query
            sql_query = f"""
            WITH all_years AS (
                {union_all_query}
            ),
            year_counts AS (
                SELECT year, COUNT(DISTINCT source_table) AS table_count
                FROM all_years
                GROUP BY year
            )
            SELECT year
            FROM year_counts
            WHERE table_count = (
                SELECT COUNT(*) FROM (VALUES {", ".join([f"('{table}')" for table in self.prefecture_name])}) AS tables(table_name)
            )
            ORDER BY year;
            """
            year_options=conn.query(sql_query)
        else:
            year_options= pd.DataFrame(data={'year':list(range(2001,2022,1))})
        year_options=year_options.astype({'year': 'int32'})
        true_step=has_stepsize_one(year_options['year'])
        year_options=year_options.astype({'year': 'str'})
        return year_options,true_step
    
    def suburbs_unioning_query(self,sub_areas):
    # Get the list of tables for the selected suburb

        required_columns = ["Date", "Hour"]
        selection_columns=','.join([f'"{col}"' for col in required_columns+self.valid_column_for_suburbs])
        # Construct the UNION ALL part of the query
        union_all_query = " UNION ALL ".join(
            f"""
            SELECT {selection_columns}
            FROM {table}
            WHERE {self.day_query} AND {self.month_query} AND {self.year_query}
            """
            for table in sub_areas
        )
        
        # Construct the final query

        return union_all_query+';'
   
    def  prefectures(self):
        subs_id_list=['THC','WTH','ETH','NTH','ACS','ANS','AWS','ASS','WAS','PIP']
        if self.region_id:
            prefecture_ids = get_values(self.region_dict,self.region_id)
            prefecture_tables = corresponding_key(prefecture_ids, self.prefecture_codenames)
            
            ### DANGE DANGER DANGER
            if self.region_id in subs_id_list or all(region_id in subs_id_list for region_id in self.region_id):
                self.prefecture_name=prefecture_tables     
            else:
                self.prefecture_name=st.multiselect("Choose from the following prefectures", prefecture_tables)
                self.prefecture_name=[name.replace(" ","_")for name in self.prefecture_name]
            
        else:
            self.prefecture_name=st.multiselect("Choose from the following prefectures",[name.replace('_',' ')for name in self.prefecture_codenames.keys()])
            self.prefecture_name=[name.replace(" ","_")for name in self.prefecture_name]
            
    def generate_dynamic_column_query(self, conn):
        """
        Generate a robust SQL query to filter data dynamically based on user inputs.    

        This function:
        1. Filters data within the specified timeframe for all selected tables.
        2. Identifies common columns across the selected tables.
        3. Checks that each common column has ≤50% null values in all tables within the timeframe.
        4. Returns valid columns for user selection.    

        Args:
            conn: Database connection object.    

        Returns:
            None: Updates self.cols with valid column names.
        """
        # Build the filtering query based on the timeframe
        filters = f"{self.day_query} AND {self.month_query} AND {self.year_query}"    

        # Step 1: Identify common columns across all selected tables
        common_columns_query = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ANY(ARRAY{[table.lower() for table in self.prefecture_name]})
            AND column_name NOT IN ('region_id', 'prefecture_id', 'Date', 'Hour')
            GROUP BY column_name
            HAVING COUNT(*) = {len(self.prefecture_name)};
        """
        common_columns = conn.query(common_columns_query)['column_name'].to_list()    

        if not common_columns:
            raise ValueError("No common columns found across the selected tables.")    

        # Step 2: Check null fractions for each common column across all tables
        #na bgei to null fraction
#        valid_columns = []
#        for column in common_columns:
#            null_fraction_checks = []
#            for table in self.prefecture_name:
#                null_fraction_query = f"""
#                    SELECT SUM(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END)::float / COUNT(*) AS null_fraction
#                    FROM {table}
#                    WHERE {filters};
#                """
#                result = conn.query(null_fraction_query).iloc[0]['null_fraction']
#                null_fraction_checks.append(result <= 0.5)    

            # Column is valid only if all checks pass
#            if all(null_fraction_checks):
#                valid_columns.append(column)    
#
#        if not valid_columns:
#            st.warning("No columns meet the criteria of less than 50% null values in all locations after filtering by timeframe.")    

        # Step 3: Update valid columns for user selection
#        self.valid_column_for_suburbs=valid_columns
        self.cols = st.sidebar.multiselect(label="Select available air pollutant",
                                           options=common_columns,help='Maximum number of air pollutants: 2',max_selections=2)
           

        return

    def fetch_simple_dataframe_info(self,conn):
        """Fetch dataframes based on the selected combinations."""
        required_columns = ["Date", "Hour"]
        selection_columns=','.join([f'"{col}"' for col in required_columns+self.cols])
        try:

            if self.prefecture_name and selection_columns:
#                if self.region_name[0] in list(self.region_names_dict.keys())[9:]:
#                    for sub in self.region_name:
#                        sub_areas=self.suburbs[sub]
#                        st.write(sub_areas)
#                        df_query=region_selector.suburbs_unioning_query(sub_areas)
#                        df = conn.query(df_query,ttl='10m')
#                        df['Hour']=pd.to_datetime(df['Hour'],format="%H:%M:%S").dt.time.astype('str')
#                        df['Date']=pd.to_datetime(df['Date'])
#                        self.dataframes[sub]=df
#                else:
                    for prefecture_name in self.prefecture_name:
                        df_query = f"""
                        SELECT {selection_columns}
                        FROM {prefecture_name}
         
                        """+"""WHERE"""+self.day_query+"""AND"""+self.month_query+"""AND"""+self.year_query+ """;"""
                        null_query=f"""SELECT * FROM {prefecture_name}"""
                        #This is the part the user sees for the usefull info he chose
                        df = conn.query(df_query,ttl='10m')
                        df['Hour']=pd.to_datetime(df['Hour'],format="%H:%M:%S").dt.time.astype('str')
                        df['Date']=pd.to_datetime(df['Date'])
                        
                        self.dataframes[prefecture_name]=df
                        
                        
                        #this is the part where we check which are the null value by goupring by per year
                        #null_dif=conn.query(null_query,ttl='10m')
                        #null_dif['Date']=pd.to_datetime(null_dif['Date'])
                        #self.null_dataframes[prefecture_name]=null_dif
                
                
                #self.keystone=st.selectbox('Choise the location you want to see the dataframe',options=self.dataframes.keys())
                #st.dataframe(self.dataframes[self.keystone])
                    #st.dataframe(df)
                #keystone=st.selectbox('Choose frame',options=self.dataframes.keys())
                #st.dataframe(self.dataframes[keystone])
                #st.write(self.dataframes.keys())
            else:
                return st.write("")

        except Exception as e:
            st.write(e)
            return None
   #SLICING PARTS OF CLASS
    
    def aggregation_method(self):
        group_columns = {
        "Year": lambda df: df['Date'].dt.year,
        "Month": lambda df: df['Date'].dt.month,
        "Day": lambda df: df['Date'].dt.date,
        'Hour':lambda df:df['Hour']}
        
        aggregated_cols={col:self.method.lower() for col in self.cols}    
        
        self.agg_dataframe={str(pref_name):pref_data.groupby(group_columns[self.timeframe](pref_data)).agg(aggregated_cols) 
                            for pref_name,pref_data in self.dataframes.items()}
        #dfs=st.button('Show dataframe aggregated')
        #if dfs:
         #   st.dataframe(self.agg_dataframe[self.keystone])
        return 
                        
    def null_graphs(self):
        df=self.null_dataframes
        key=self.keystone
        null_percentage = (df[key].iloc[:,:-2].groupby(df[key]['Date'].dt.year)
                           .apply(lambda group: group.isnull().mean() * 100)
                           .drop(columns='Date')  # Drop the 'year' column from the result
)
        
        st.bar_chart(null_percentage,stack=False)
        
        #df['year']=df['Date'].dt.year
        
    def dynamic_groupby_aggregation(self):
        """ Dynamically create a grouped aggregation graph based on user selections.
        Handles one or two gases and multiple locations."""
        #ngases=len(self.cols)
        #ncities=len(list(self.dataframes.keys()))
        #st.write(ncities)
        # Plot the graph
        locations=[name.replace("_"," ")for name in self.prefecture_name][0]
        #st.markdown("<h2 style='text-align: center; color: grey;'>Big headline</h2>", unsafe_allow_html=True)
        #st.subheader(f"Aggregated Data Visualization of station {locations}\t")
        fig = go.Figure()    
        
        offsetgroup = 0
        #st.dataframe[self.agg_dataframe['BOLOS']]
        for idx,col in enumerate(self.cols):
            for location, agg_data in self.agg_dataframe.items():
                fig.add_trace(
                    go.Bar(
                        x=agg_data.index,
                        y=agg_data[col],
                        name=f"{location} - {col}",
                        offsetgroup=str(offsetgroup),
                        marker=dict(opacity=0.8),
                        yaxis="y" if idx == 0 else "y2"
                    )
                )
                offsetgroup += 1
        layout_args = { "barmode": "group",  # Ensures bars are placed side-by-side
                        "xaxis": {"title": "Date"},
                        "legend_title": "Location - Gas",
                        "legend": {
        "x": 1.04,  # Right side (0=left, 1=right)
        "y": 1,  # Top (0=bottom, 1=top)
        "xanchor": "left",  # Anchor point for x position
        "yanchor": "top"},"margin": {
            "l": 60,
            "r": 150,  # Dynamic right margin
            "b": 60,
            "t": 60,
            "pad": 4
        }   # Anchor point for y position
                        }
        if len(self.cols) == 1:
            layout_args["yaxis"] = {"title": self.cols[0]}# Single Y-axis case
            layout_args["annotations"] = [
            dict(
            x=0.5,  # Center the text
            y=1.20,  # Slightly above the plot
            xref="paper",
            yref="paper",
            text=f"Pollution Levels of station {locations} <br>for {self.cols[0]} gas",
            showarrow=False,
            font=dict(size=25),
            align="center"
        )
    ]
            layout_args["title"] = ''
            
        else:
            locations=' , '.join([name.replace("_"," ")for name in self.prefecture_name])
            layout_args["yaxis"] = {"title": self.cols[0]}
            layout_args["title"]=f"Pollution Levels of stations {locations} for {self.cols[0]}"
            layout_args["yaxis2"] = {
                     "title": self.cols[1],
                     "overlaying": "y",  # Overlay on primary y-axis
                     "side": "right",
                     "showgrid": False
                                    }

            layout_args["annotations"] = [
        dict(
            x=0.5,  # Center the text
            y=1.20,  # Slightly above the plot
            xref="paper",
            yref="paper",
            text=f"Pollution Levels of stations {locations} <br>for {self.cols[0]}",
            showarrow=False,
            font=dict(size=25),
            align="center"
        )
    ]
            layout_args["title"] = ''
        fig.update_layout(**layout_args)
        fig.update_layout( title_automargin=True,  # Forces auto-adjustment
    autosize=True          # Ensures proper container sizing
)
        
        st.plotly_chart(fig, use_container_width=True)
         
    
    def year_month_day_selection_query(self,conn):
        #we are using dictionaries for months and days as they are stable in contrast to years
        #that is why we are using sliders
        #there is a possibility in the future to use instead of range the multiselec boxes
        months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
        ]
        
        days =['Monday','Tuesday','Wednesday','Thursday','Firday','Saturday','Sunday']
        month_map = {month: index + 1 for index, month in enumerate(months)}
        days_map={day: index + 1 for index, day in enumerate(days)}
        month_slider_values = st.sidebar.select_slider(label=' ',options=months,value=("January", "December"),help='Here you can set the range \n of months you would like to apply to the data') 
                                         # Default to the full range of months

        day_slider_values=st.sidebar.select_slider(" ",options=days,value=('Monday','Sunday'),help='Select days range (for just one day get both sides of slider at the day you want)')
        # Capture the start and end month names from the slider
        #st.sidebar.select_slider('Choose avalable years',options=list(conn.query(region_selector.generate_dynamic_sql())))
        year_options,true_step=region_selector.common_year_selection(conn)
        
        #Bazoume edo to eidos tou aggregation katho kai episi os pros ti tha kanoume omadopoihsh
        self.method=st.sidebar.selectbox(' ',['Mean','Median',],help="Choose aggregation method")
        self.timeframe=st.sidebar.selectbox(' ',['Year','Month','Day','Hour'],help="Choose timeframe for aggregation")
        
        #Here we have the year option with three options the frist is when we have one year only the second is if we have continuous 
        #yeas and the third is when we have not continuous years
        if year_options.size<2:
            st.write(f"Available year only {year_options['year'].iloc[0]}")
            chosen_years=year_options['year'][0]
            self.year_list=chosen_years
        else:
            if true_step:
                self.year_list=st.sidebar.slider(label=" ", min_value=int(year_options['year'].iloc[0]),max_value=int(year_options['year'].iloc[-1]),
                                              value=(int(year_options['year'].iloc[0]), int(year_options['year'].iloc[-1])),help="Select years range (for just one year get both sides of slider at the year you want)")
                self.year_list=list(range(self.year_list[0], self.year_list[-1]+1))
                self.year_list = ', '.join(map(str, self.year_list))
            else:
                self.year_list=st.sidebar.multiselect('Choose from available years',year_options)
                self.year_list = ', '.join(map(str, self.year_list))
        #here we state the starting and ending days as well as months
        start_month, end_month = month_slider_values
        start_day,end_day=day_slider_values
        # Convert month names to numbers
        start_month_num = month_map[start_month]
        end_month_num = month_map[end_month]
        start_day_num=days_map[start_day]
        end_day_num=days_map[end_day]
        #Here is the query for the timeframes we want
        self.day_query=f"""(extract(ISODOW from "Date") between {start_day_num} and {end_day_num})"""
        self.month_query=f"""(extract(Month from "Date") between {start_month_num} and {end_month_num})"""
        self.year_query=f"""(extract(Year from "Date") IN ({self.year_list}))"""
        return None 




conn = st.connection("neon",type="sql")
region_selector= User_selection('selection_1', region_names_dict, region_dict,prefecture_codenames,suburbs)

st.title("GAS APP")
st.sidebar.title('Time and gas filters')



region_selector.regions_selection()
region_selector.prefectures()

region_selector.year_month_day_selection_query(conn)

#run_query=st.sidebar.button('Run query')
if is_query_complete(region_selector.prefecture_name,region_selector.year_list,region_selector.region_id):
     region_selector.generate_dynamic_column_query(conn)
     region_selector.fetch_simple_dataframe_info(conn)
    # Allow column queries and graph rendering
     if column_check(region_selector.cols):
        region_selector.aggregation_method()
        checker=st.checkbox("Show graph",value=False)
        if checker:
            region_selector.dynamic_groupby_aggregation()
            #region_selector.null_graphs()
     else:
        st.info('Please select at least one 1 air pollutant')

















