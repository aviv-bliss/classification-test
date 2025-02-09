# README

## Classification of generated-source image pairs

Classification of generated images compared to their source images


### Training
To train the model and to run validation in parallel, run the following command:

`python main.py config/classification.json`

To train the model, run the following command:

`python main.py config/classification.json -m train`

To run validation in parallel, run the following command:

`python main.py config/classification.json -m test`

### Testing
To test the model, run the following command:

`python main.py config/classification.json -m val`

### Inference
To run an inference the model, run the following command:

`python inference.py /mnt/disk1/models/classification/2502Feb04_18-57-08_classification_l0.0001/ckpts/model_ckpt_10_400.pth.tar -i /mnt/disk1/data/generated_dataset/input_images/7041.png -o /mnt/disk1/data/generated_dataset/output_images/7041.png -s /home/victoria/Desktop/results`
