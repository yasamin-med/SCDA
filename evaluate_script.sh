python evaluate.py --data_path "/home/yasamin/Documents/Data_breast/Augmented_0.5_percent/colorful"\
 --data_test_path "/home/yasamin/Documents/Data_breast/Original/test"\
 --data_valid_path "/home/yasamin/Documents/Data_breast/Original/validation"\
 --output_path "/home/yasamin/Documents/finetune-sd/Breast/results_color"\
 --adjective_list ""\
 --adjective_flag 0\
 --batch_size 32\
 --num_class 3\
 --num_epochs 100\
 --train 1\
 --name_result_file "result_colorful.txt"\
 --baselines 'squeezenet1.1'

