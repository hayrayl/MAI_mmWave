### simple code to store any miscellaneous functions we might have
##
import json

def read_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data