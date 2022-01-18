#python3 test.py --sample_csv ../food_data/testcase/sample_submission_main_track.csv --output_csv output_main_track.csv --model_path ./final_multitask.bin
python3 test.py --sample_csv $1 --output_csv $2 --model_path $3 
