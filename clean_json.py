#!/usr/bin/env python
# 
# Reformat Yelp data to valid json file

import json
import codecs

DATA_FILE = "data/yelp_academic_dataset_business.json"

def open_data_file(filename):
    return open(filename, "r")

def main():
    f = open_data_file(DATA_FILE)
    data = []
    with codecs.open("data/valid_json.json", "rU", "utf-8") as valid:
        for line in f:
            data.append(json.loads(line))
    f.close()

    with open("data/valid_json.json", "w") as valid:
        json.dump(data, valid)

if __name__ == "__main__":
    main()
