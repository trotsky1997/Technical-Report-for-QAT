#!/usr/bin/env python3.7
# coding: utf-8
import os

for data in ["mnist","fashionmnist","kmnist"]:
    for net in ["shufflenet","resnet","mobilenet"]:
        os.system(f"python3.7 ./main.py {data} {net}")
