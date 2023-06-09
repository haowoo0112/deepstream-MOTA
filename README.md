# deepstream-MOTA  


## how to use  

```
python main.py --GT_path elementary_school.csv --data_path data/ele/ --remove_percent 0
```

--GT_path: the path of ground truth
|Position |Name |Description
|-|-|-
|1|Frame number|Indicate at which frame the object is present
|2|Identity number|Each pedestrian trajectory is identified by a unique ID (−1 for detections)
|3|Bounding box left|Coordinate of the top-left corner of the pedestrian bounding box
|4|Bounding box top|Coordinate of the top-left corner of the pedestrian bounding box
|5|Bounding box width|Width in pixels of the pedestrian bounding box
|6|Bounding box height |Height in pixels of the pedestrian bounding box
|7|Confidence score |Indicates how confident the detector is that this instance is a pedestrian. For the ground truth and results, it acts as a flag whether the entry is to be considered.
|8|x |3D x position of the pedestrian in real-world coordinates (−1 if not available)
|9|y |3D y position of the pedestrian in real-world coordinates (−1 if not available)
|10|z|3D z position of the pedestrian in real-world coordinates (−1 if not available)

--data_path: the path of deepstream output data  
("%s %lu 0.0 0 0.0 %f %f %f %f 0.0 0.0 0.0 0.0 0.0 0.0 0.0 %f\n", obj->obj_label, id, left, top, right, bottom, confidence)