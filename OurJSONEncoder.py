#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 19:38:45 2023

@author: ownjo
"""

from json import JSONEncoder
import numpy as np

class OurJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return obj.item()
        if isinstance(obj, np.float32):
            return obj.item()
        return super().default(obj)