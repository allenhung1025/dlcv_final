import os
import re
import json
import matplotlib.pyplot as plt
from PIL import Image

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def out_json(data, out_path, cover=True):
    if check_exist(out_path) and cover == False: return
    with open(out_path, 'w') as outfile: json.dump(data, outfile)

def check_exist(out_path):    
    if re.compile(r'^.*\.[^\\]+$').search(out_path):
        out_path = os.path.split(out_path)[0]        
    existed = os.path.exists(out_path)
    if not existed:
        os.makedirs(out_path, exist_ok=True)
    return existed