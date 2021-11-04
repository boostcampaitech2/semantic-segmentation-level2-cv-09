pip install -r requirements.txt
python get_coco_mask.py --input_dir ../../input/data/ --split train_all
python copy_paste.py --input_dir ../../input/data/ --output_dir ../../input/data/ --patch ["Paper pack", "Battery", "Plastic", 'Clothing',"Glass" ] --remove_patch ["Paper", "Plastic bag"] --json_path train.json --lsj True --lsj_max 2 --lsj_min 0.2 --aug_num 1500 --extract_patch True 
python mask_coco_mask.py --main_json train.json --mode add