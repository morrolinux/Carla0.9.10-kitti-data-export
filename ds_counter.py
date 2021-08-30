#!/usr/bin/env python3

import json

ds_counter = {"Cyclist": 0, "Car": 0, "Pedestrian": 0}

with open("_out/training/dslog.json", 'r') as f:
    dslog = json.load(f)

for weather in dslog:
    for location in weather["locations"]:
        for ds_class, qty in location["ds_classes"].items():
            ds_counter[ds_class] += qty

print(ds_counter)
