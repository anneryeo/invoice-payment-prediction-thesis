import pandas as pd
import json
import os

class Settings:
    def __init__(self):
        config = self.get_configs()
    
    def get_configs(self):
        settings = {}
        
        f = open('settings.json')
        json_f = json.load(f)
        df_json_directory = pd.DataFrame(json_f['Root Folders'])
        df_json_settings = pd.DataFrame(json_f['Config'])
        
        current_user = os.getlogin()
        df_json = df_json_directory[df_json_directory['user'] == current_user]
        df_json.reset_index(inplace=True)

        self.database_folder = df_json_directory['databases'][0]
        self.install_folder = df_json_directory['install_folder'][0]
        
        settings['if_debug_mode'] = df_json_settings['debug_mode'][0] == "True"
    
        return settings

    def get_sub_directories(self):
        directory = {}

        directory['install_folder'] = self.install_folder
        directory['revenues_folder'] = f"{self.database_folder}/Revenues and Expenses"
        directory['student_file'] = f"{self.database_folder}/Students.xlsx"
        directory['file_location_settings'] = f"{self.database_folder}/settings.xlsx"
        

        
        return directory