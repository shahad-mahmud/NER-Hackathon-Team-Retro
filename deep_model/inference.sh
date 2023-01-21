#for single sentence

# python inference.py -fs sentence \
#                     --is_banner \
#                     --weighted \
#                     --best_model_index 1 \
#                     --pretrained_path model_path \
#                     --sen "২৮ সেপ্টেম্বর দুবাইতে অনুষ্ঠিত হবে এই ১৫ সেপ্টেম্বর শ্রীলঙ্কার বিপক্ষে এবং ২০ সেপ্টেম্বর আফগানিস্তানের বিপক্ষে খেলবে"

#for multiple sentences, send a file

python inference.py -fs file \
                    --data_dir text_file_path \
                    --output_dir output_folder \
                    --is_banner \
                    --weighted \
                    --best_model_index 1 \
                    --pretrained_path model_path
