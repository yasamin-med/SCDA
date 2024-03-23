python evaluate_new.py --data_path "/home/yasamin/Documents/Data_breast/Original"\
 --data_test_path "/home/yasamin/Documents/Data_breast/miccai_breast"\
 --data_valid_path "/home/yasamin/Documents/Data_breast/Original/validation"\
 --output_path "/home/yasamin/Documents/finetune-sd/Breast/results_original"\
 --adjective_list "" \
 --baselines 'densenet121','resnet34','squeezenet1.1'\
 --adjective_flag 0\
 --batch_size 32\
 --num_class 3\
 --num_epochs 100\
 --train 0\
 --output_file_name "result_miccai"\
 --size 224
 #"bright","colorful","dark","high-contrast","low-contrast","no_adjective","posterized","sheared","solarized","stylized"

