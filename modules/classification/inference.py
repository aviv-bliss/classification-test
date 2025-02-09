import numpy as np
import argparse
import json
import os
import time
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
import PIL


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = '1'
import dataloaders.img_transforms as transforms
from utils.auxiliary import load_model_and_weights, softmax, resize_and_fill

import torch
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

'''
    Inference on a single/multiple images.
    Input: ckpt path, image/dir path.
    Loads test config, runs inference on an image or images in a given directory.
    Since model is fully convolutional, it can run on an image of any size. 
    
    Example:
        Single image inference:
            python inference.py /mnt/disk1/models/classification/2502Feb04_18-57-08_classification_l0.0001/ckpts/model_ckpt_10_400.pth.tar
            -i /mnt/disk1/data/generated_dataset/input_images/7041.png
            -o /mnt/disk1/data/generated_dataset/output_images/7041.png
            -s /mnt/disk1/models/classification/2502Feb03_11-18-22_classification_n34/results
            
        Inference of all images in the given directory:
            python inference.py /mnt/disk1/models/classification/2502Feb04_18-57-08_classification_l0.0001/ckpts/model_ckpt_10_400.pth.tar
            -id /mnt/disk1/data/generated_dataset/input_images
            -od /mnt/disk1/data/generated_dataset/output_images
            -s  /mnt/disk1/models/classification/2502Feb03_11-18-22_classification_n34/results
'''

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', metavar='DIR', help="Path to a model ckpt")
    parser.add_argument('-i', '--input_image_path', type=str, default='',
                        help="Path to an input (original) image (for a single image inference)")
    parser.add_argument('-o', '--output_image_path', type=str, default='',
                        help="Path to an output (generated) image (for a single image inference)")
    parser.add_argument('-id', '--input_image_dir', type=str, default='',
                        help="Path to an input (original) image directory (for an inference of all images in the given directory)")
    parser.add_argument('-od', '--output_image_dir', type=str, default='',
                        help="Path to an output (generated) image directory (for an inference of all images in the given directory)")
    parser.add_argument('-s', '--save_dir', type=str, default='',
                        help="(optional) Path to a directory to save resulted images")
    parser.add_argument('--show', type=bool, default=False, help="Show resulted images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU number. If '-1', runs on CPU")
    args = parser.parse_args()
    return args


def load_model(model_path, gpu=0):
    model_path = Path(model_path)
    train_dir = model_path.parent.parent
    config_path = train_dir / 'args_test.json'
    assert os.path.isfile(config_path)
    with open(config_path, 'rt') as r:
        config = json.load(r)

    class FLAGS():
        def __init__(self):
            self.load_ckpt = model_path
            self.batch_size = config['batch_size']
            self.train_dir = train_dir
            self.height = config['height']
            self.width = config['width']
            self.data_loader = config['data_loader']
            self.model = config['model']
            self.metric = config['metric']
            self.num_channels = config['num_channels']
            self.num_classes = config['num_classes']
            self.batch_norm = config['batch_norm']
            self.version = config['version']
            self.num_layers = config['num_layers']
            self.pretrained = config['pretrained']
            self.fc = config['fc']
            self.seed = config['seed']
    FLAGS = FLAGS()

    # run on GPU/CPU
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # print('running inference on GPU {}'.format(gpu))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''  # run on cpu
        print('running on CPU')

    # load model
    models_loaded, models, model_names, n_iter, n_epoch = load_model_and_weights(model_path, FLAGS, use_cuda,
                            model=config['model'], ckpts_dir=train_dir/'ckpts', train=False)
    model = models[0]

    return models_loaded, model


def resize_and_fill_black(img1, img2, resize_h, resize_w):
    if img1.size != img2.size:
        if img1.size[0] > img2.size[0] or img1.size[1] > img2.size[1]:
            img1 = img1.resize(img2.size, PIL.Image.ANTIALIAS)
        else:
            img2 = img2.resize(img1.size, PIL.Image.ANTIALIAS)
    assert img1.size == img2.size

    w, h = img1.size
    if w == resize_w and h == resize_h:
        return img1, img2

    img1_resized = resize_and_fill(img1, resize_w, resize_h)
    img2_resized = resize_and_fill(img2, resize_w, resize_h)

    return img1_resized, img2_resized


def inference(model, input_img, output_img):
    '''
    :param model_path: path to ckpt
    :param input image: np array of shape (B, H, W, C)
    :param output image: np array of shape (B, H, W, C)
    '''
    if len(input_img.shape) == 3 and len(output_img.shape) == 3:
        input_img = input_img.reshape(-1, input_img.shape[0], input_img.shape[1], input_img.shape[2])
        output_img = output_img.reshape(-1, output_img.shape[0], output_img.shape[1], output_img.shape[2])
    else:
        print('Input or/and output image shapes are not correct')

    model.eval()

    # transform to pytorch tensor
    totensor = transforms.Compose([transforms.ToTensor(), ])
    input_img_t = totensor(input_img)          # (batch_size, num_channels, height, width)
    output_img_t = totensor(output_img)        # (batch_size, num_channels, height, width)
    input_img_t = Variable(input_img_t)
    output_img_t = Variable(output_img_t)
    if use_cuda:
        input_img_t = input_img_t.cuda()
        output_img_t = output_img_t.cuda()

    # get model predictions
    output = model(input_img_t, output_img_t)
    logits = output.data.cpu().numpy()[0]
    probability = softmax(logits)  # shape [N, n]
    pred_score = np.argmax(probability)

    return pred_score


def result_visualization(img1, img2, text, filename, save_dir, show=False):
    '''
    :param img1: PIL image
    :param img2: PIL image
    :param filename: filename including extension
    '''
    # Calculate the total width and the maximum height for the concatenated image
    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)

    # Create a new blank image with a white background
    new_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

    # Paste the images side by side
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    # Create a drawing context on the new image
    draw = ImageDraw.Draw(new_img)

    # Optional: load a TrueType font. Update the font path and size as needed.
    font = ImageFont.load_default()

    # # Calculate text size to help with positioning (e.g., centering at the bottom)
    # text_width, text_height = draw.textsize(text, font=font)
    # Calculate text bounding box and dimensions for positioning
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]  # right - left
    text_height = bbox[3] - bbox[1]  # bottom - top

    # Example: position the text at the bottom center of the concatenated image with a margin
    x = (total_width - text_width) // 2
    y = max_height - text_height - 10  # 10-pixel margin from the bottom

    # Choose a text color (R, G, B)
    text_color = (100, 250, 50)

    # Draw the text on the image
    draw.text((x, y), text, fill=text_color, font=font)

    # Save the concatenated image with the text overlay
    if save_dir != '':
        output_path = os.path.join(save_dir, filename)
        new_img.save(output_path)
        print(f"Concatenated image saved to {output_path}\n")

    if show:
        new_img.show()


def inference_and_visualization(input_image_path, output_image_path, model, resize_h=768, resize_w=512, save_dir='',
                                show=False):
    input_img = Image.open(input_image_path)
    output_img = Image.open(output_image_path)
    resize_and_fill_black(input_img, output_img, resize_h, resize_w)
    # input_img = input_img.resize((768, 512), PIL.Image.LANCZOS)
    input_img_np = np.array(input_img)
    # output_img = output_img.resize((768, 512), PIL.Image.LANCZOS)
    output_img_np = np.array(output_img)

    # run inference
    pred_score = inference(model, input_img_np, output_img_np)

    # visualize
    if save_dir != '' or show:
        if save_dir != '' and not os.path.exists(save_dir):
            os.mkdir(save_dir)

        filename = input_image_path.split('/')[-1]
        text = f'Predicted score: {pred_score}'
        result_visualization(input_img, output_img, text, filename, save_dir, show=show)

    return pred_score


# def main(model_path, input_image_path, output_image_path, gpu=0, save_dir='', show=False, all_dir=False):
def main(args=None):
    t_i = time.time()
    if args is None:
        args = getArgs()
    model_path = args.model_path
    all_dir = False
    if args.input_image_path and args.output_image_path:
        input_image_path = args.input_image_path
        output_image_path = args.output_image_path
    elif args.input_image_dir and args.output_image_dir:
        input_image_path = args.input_image_dir
        output_image_path = args.output_image_dir
        all_dir = True
    else:
        print('Input or/and output image paths are not correct')

    gpu = args.gpu
    save_dir = args.save_dir
    show = args.show

    if save_dir != '' and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load model and run inference
    models_loaded, model = load_model(model_path, gpu=gpu)
    if models_loaded:
        if not all_dir:
            if input_image_path != '' and output_image_path != '' and os.path.isfile(input_image_path) and os.path.isfile(output_image_path):
                pred_score = inference_and_visualization(input_image_path, output_image_path, model, save_dir=save_dir, show=show)

                if save_dir != '' or show:
                    print('run time: {:.2f} min'.format((time.time() - t_i) / 60))

                return pred_score
            else:
                print('Input or/and output image paths are not correct')
                exit()
        else:
            pred_score_lst = []
            input_image_lst = [filename for filename in os.listdir(input_image_path) if filename.split('.')[-1] in ['jpg', 'png']]
            output_image_lst = [filename for filename in os.listdir(output_image_path) if filename.split('.')[-1] in ['jpg', 'png']]
            assert len(input_image_lst) == len(output_image_lst), 'Number of input and output images are not equal'

            for filename in input_image_lst:
                input_img_path = os.path.join(input_image_path,filename)
                output_image_path =  os.path.join(output_image_path, filename)
                if os.path.exists(input_img_path) and os.path.exists(output_image_path):
                    pred_score = inference_and_visualization(input_img_path, output_image_path, model, save_dir=save_dir)
                    pred_score_lst.append(pred_score)

            if save_dir != '' or show:
                print('run time: {:.2f} min'.format((time.time() - t_i) / 60))
            return pred_score_lst
    else:
        print('Model did not load')
        exit()




if __name__ == '__main__':
    main()















