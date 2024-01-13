### we manually chose 5 table from objaverse.
### The table can be chosen automatically from objaverse in the future, just as like other objects.

import numpy as np
table_paths = [
    "table", 
    "table_1d43b7612af94e1183f00e604d9edf4a", 
    "table_9c060629c8a04b038f562977317f6cb1", 
    "table_57013651506042f885fad59a4e0994a9",
    "table_fb0efe334cc046e383c0f21fc70ed82b"
]

# this is manually tuned for now, but can be automatically determined, as like other objects. 
table_scales = {
    "table": 0.7,
    "table_1d43b7612af94e1183f00e604d9edf4a": 2.5,
    "table_9c060629c8a04b038f562977317f6cb1": 1.8,
    "table_57013651506042f885fad59a4e0994a9": 1.9,
    "table_fb0efe334cc046e383c0f21fc70ed82b": 0.8,
}

table_poses = {
    "table": [0, 0, 0],
    "table_1d43b7612af94e1183f00e604d9edf4a": [0, 0, 0.195],
    "table_9c060629c8a04b038f562977317f6cb1": [0, 0, 0.13],
    "table_57013651506042f885fad59a4e0994a9": [0, 0, 0.2],
    "table_fb0efe334cc046e383c0f21fc70ed82b": [0, 0, 0],
}

# this is to handle the case where pybullet gives very wrong bounding box for the table (much larger than the actual table size)
# might be due to outlier points in the table mesh -- not sure how exactly pybullet processes the mesh to get the bounding box
# interestlying this seems to be only happening for the tables, not other objects
table_bbox_scale_down_factors = {
    "table": 0.1,
    "table_1d43b7612af94e1183f00e604d9edf4a": 0.35,
    "table_9c060629c8a04b038f562977317f6cb1": 0.35,
    "table_57013651506042f885fad59a4e0994a9": 0.35,
    "table_fb0efe334cc046e383c0f21fc70ed82b": 0.05,
}