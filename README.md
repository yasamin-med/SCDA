

<h1> MEDDAP: Medical Dataset Enhancement via Diversified Augmentation Pipeline </h1>
<h3>
Yasamin Medghalchi, Niloufar Zakariaei, Arman Rahmim, and Ilker Hacihaliloglu </h3>



The effectiveness of Deep Neural Networks (DNNs) heavily relies on the abundance and accuracy of available training data. However, collecting and annotating data on a large scale is often both costly and time-intensive, particularly in medical cases where practitioners are already occupied with their duties. Moreover, ensuring that the model remains robust across various scenarios of image capture is crucial in medical domains, especially when dealing with ultrasound images that vary based on the settings of different devices and the manual operation of the transducer.

To address this challenge, we introduce a novel pipeline called MEDDAP, which leverages Stable Diffusion (SD) models to augment existing small datasets by automatically generating new informative labeled samples. Pretrained checkpoints for SD are typically based on natural images, and training them for medical images requires significant GPU resources due to their heavy parameters. To overcome this challenge, we introduce USLoRA (Ultrasound Low-Rank Adaptation), a novel fine-tuning method tailored specifically for ultrasound applications. USLoRA allows for selective fine-tuning of weights within SD, requiring fewer than 0.1% of parameters compared to fully fine-tuning only the UNet portion of SD.

To enhance dataset diversity, we incorporate different adjectives into the generation process prompts, thereby desensitizing the classifiers to intensity changes across different images. This approach is inspired by clinicians' decision-making processes regarding breast tumors, where tumor shape often plays a more crucial role than intensity. In conclusion, our pipeline not only outperforms classifiers trained on the original dataset but also demonstrates superior performance when encountering unseen datasets.

## Method Pipeline
Please refer to the [paper](https://arxiv.org/pdf/2403.16335.pdf) for more technical details.

## Requirements

* Install necessary python libraries:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
cd examples/text_to_image
pip install -r requirements.txt
```
```bash
accelerate config -----> set the config based on your device
```
create account on hugging face and copy the access token for writing mode and paste the token when it is asked
```bash
pip install --upgrade huggingface_hub
huggingface-cli login 
```
## Prepairing Dataset
Make Dataset in Hugging face webpage in your account with your preferred name
Run the below code in terminal
```bash
python make_dataset_breast.py --train_dir <path_to_train_directory>\
 --classes 'benign','malignant','normal'\
 --prompt_structure "an ultrasound photo of {class_name} tumor in breast"\
 --dataset_path <path_to_dataset_hugging_face>\
 --token <writing_token_hugging_face>
```
Remmember if you want to use this pipeline for different application, in the code, I change the structure of prompt for "normal' class. Instead of writing "an ultrasound photo of normal tumor in breast", I wrote "an ultrasound photo of no tumor in breast". This happens in "Make New Expanded Datasets" section too.
## Fine-tuning with USLoRA
you can chnge the hyperparameters based on your problem but remember that it works better with batch size equal to 1.
```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME=<path_to_dataset_hugging_face>
accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=224 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir=<path_of_checkpoints> \
  --validation_prompt="an ultrasound photo of benign tumor in breast" --report_to="wandb" --rank 4
```
## Make New Expanded Datasets
In this section, we aim to generate new datasets with different adjectives and expansion ratios.

```bash
python Inference_percent_breast.py --model_path <path_of_checkpoints>\
  --adjective_list "","colorful","stylized","high-contrast","low-contrast","posterized","solarized","sheared","bright","dark" \
  --modes "benign","malignant","normal"\
  --percent_list 0.5,1.0,2.0\
  --prompt_structure "an ultrasound photo of {class_name} tumor in breast"\
  --existing_images_base_directory <path_of_existing_original_dataset_local>\
  --save_dir <path_of_expanded_dataset>\
  --copy_flag 0
```
put the name of classes of your dataset in "mode" and the expansion ration in "percent_list". Remmember to not put any space between inputs seperating with cammas. "copy_flag" equal to 1 means if you want to copy the original images in new dataset as well, otherwise your new dataset contains just synthetic images.

## Training and evaluation for downstream task, classification
In this section, we aim to train classifiers for both original and mixed (original + synthetic) ones. we used  
'densenet121','resnet34','squeezenet1.1' as classifiers but we wrote a code for other classifiers in the code, you can choose them for your project but pay attention to change their last layer to work best with your problem.
```bash
python evaluate_new.py --data_path <dir_of_dataset_local>\
 --data_test_path <path_of_test_dataset>\
 --data_valid_path <path_of_validation_set>\
 --output_path <dir_of_output>\
 --adjective_list "bright","colorful","dark","high-contrast","low-contrast","no_adjective","posterized","sheared","solarized","stylized" \
 --baselines 'densenet121','resnet34','squeezenet1.1'\
 --adjective_flag 0\
 --batch_size 32\
 --num_class 3\
 --num_epochs 100\
 --train 0\
 --output_file_name <name_of_result_text_and_table>\
 --size 224
```
### Citation
```
@misc{medghalchi2024meddap,
      title={MEDDAP: Medical Dataset Enhancement via Diversified Augmentation Pipeline}, 
      author={Yasamin Medghalchi and Niloufar Zakariaei and Arman Rahmim and Ilker Hacihaliloglu},
      year={2024},
      eprint={2403.16335},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
