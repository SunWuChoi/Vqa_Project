# visual_question_answering
Pytorch implementation of the following papers 
- VQA: Visual Question Answering (https://arxiv.org/pdf/1505.00468.pdf).
- Stacked Attention Networks for Image Question Answering (https://arxiv.org/abs/1511.02274)
- Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering (https://arxiv.org/abs/1612.00837)
- Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering (https://arxiv.org/abs/1707.07998)


![model](./png/basic_model.png)

## Directory and File Structure
```
.
+-- datasets/
|   +-- images/
|       +-- test2015/
|           +-- COCO_test2015_000000000001.jpg
|           +-- COCO_test2015_000000000014.jpg
|           +-- ...
|       +-- train2014/
|           +-- COCO_train2014_000000000009.jpg
|           +-- COCO_train2014_000000000025.jpg
|           +-- ...
|       +-- val2014/
|           +-- COCO_val2014_000000000042.jpg
|           +-- COCO_val2014_000000000073.jpg
|           +-- ...
|   +-- Questions/
|       +-- v2_OpenEnded_mscoco_test2015_questions.json
|       +-- v2_OpenEnded_mscoco_test-dev2015_questions.json
|       +-- v2_OpenEnded_mscoco_train2014_questions.json
|       +-- v2_OpenEnded_mscoco_val2014_questions.json
|   +-- Annotations/
|       +-- v2_mscoco_train2014_annotations.json
|       +-- v2_mscoco_val2014_annotations.json
|   +-- test.npy
|   +-- test-dev.npy
|   +-- train.npy
|   +-- train_valid.npy
|   +-- valid.npy
|   +-- vocab_answers.txt
|   +-- vocab_questions.txt
```


## Usage 

#### 1. Clone the repositories.
```bash
$ git clone https://github.com/SatyamGaba/visual_question_answering.git
```

#### 2. Download and unzip the dataset from official url of VQA: https://visualqa.org/download.html.
We have used VQA2 in for this project
```bash
$ cd visual_question_answering/utils
$ chmod +x download_and_unzip_datasets.csh
$ ./download_and_unzip_datasets.csh
```

#### 3. Preproccess input data for (images, questions and answers).

```bash
$ python resize_images.py --input_dir='../COCO-2015/Images' --output_dir='../COCO-2015/Resized_Images'  
$ python make_vacabs_for_questions_answers.py --input_dir='../COCO-2015'
$ python build_vqa_inputs.py --input_dir='../COCO-2015' --output_dir='../COCO-2015'
```

#### 4. Train model for VQA task.

```bash
$ cd ..
$ python train.py --model_name="<name to save logs>" --resume_epoch="<epoch number to resume from>" --saved_model="<saved model if resume training>"
```
#### 5. Plotting.
Rename model_name variable in `plotter.py`
```bash
$ python plotter.py
```

#### 6. Infer the trained model on an Image.

```bash
$ python test.py --saved_model="<path to model>" --image_path="<path to image>" --question="<ask question here>"
```

## References
* Paper implementation
  + Keywords: Visual Question Answering ; Simple Attention; Stacked Attention; Top-Down Attention;
    
* Baseline Model
  + Github: https://github.com/tbmoon/basic_vqa
