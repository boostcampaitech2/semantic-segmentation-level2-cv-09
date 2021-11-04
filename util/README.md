# Oversampling
## Example
```
# Options
--patch # target patch class ex) Battery
--background # target background class ex) "Metal" "Glass" "General trash"
--num_output # output number ex) 500
--json_path # path of coco dataset json file 
--start_index # starting number of image file name ex) 60000 -> 60000.jpg
--min_area # min size of patch 
--max_area # max size of patch
```

```
python oversampling.py --path Battery --num_output 1000
# generate 1000 oversampled image (Battery) 
```
