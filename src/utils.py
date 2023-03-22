import os
import sys
import numpy as np
import pandas as pd
import dill # this library help us to create the pickle file


sys.path.insert(0, 'D:\Storage\mlproject\src') #sys.path.append('D:\Storage\mlproject\src') 

from exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)