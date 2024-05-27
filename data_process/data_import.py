import numpy as np
from pathlib import Path
import laspy
import tifffile

class DataImporter:
    def __init__(self) -> None:
        self.data_list =[]

    def import_txt_data(self, txt_file_path, data_type="int"):
        if txt_file_path.is_file() and txt_file_path.suffix==".txt":
            data_list = []
            with open(txt_file_path, 'r') as file:
                lines = file.readlines()
            for line in lines:
                items = line.strip().split()
                if data_type=="int":
                    data = [int(float(item)) for item in items]
                    data_list.append(data)
                elif data_type=="float":
                    data = [float(item) for item in items]
                    data_list.append(data)
                else:
                    data = [str(item) for item in items]
                    data_list.append(data)
            self.data_list.append(data_list)
        else:
            print(f"Invalid txt file: {txt_file_path}")

    def import_las_data(self, las_file_path, position=True, colors=False):
        if las_file_path.is_file and las_file_path.suffix==".las":
            cloud = laspy.read(las_file_path)
            if not position and not colors:
                print(f"Import ploudcloud feature error")
                return
            if position:
                points = cloud.xyz
            if colors:
                pass
            self.data_list.append(points)

        else:
            print(f"Invalid LAS file: {las_file_path}")
            return
        
    def import_tif_data(self, tif_file_path):
        if tif_file_path.is_file() and tif_file_path.suffix==".las":
            img = tifffile.imread(tif_file_path)
            self.data_list.append(img)
        else:
            print(f"Invalid LAS file: {tif_file_path}")
            return 
        
    def import_data(self, file_paths, data_type=[]):
        for i, file_path in enumerate(file_paths):
            if file_path.suffix==".txt":
                self.import_txt_data(file_path, data_type[i])
            elif file_path.suffix==".las":
                self.import_las_data(file_path)
            elif file_path.suffix==".tif":
                self.import_tif_data(file_path)
            else:
                print(f"Unsupported file format: {file_path}")

    def get_data(self):
        return self.data_list


