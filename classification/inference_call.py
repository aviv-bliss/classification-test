import argparse
from inference import main





if __name__ == '__main__':
    args = argparse.Namespace(
        model_path='/mnt/disk1/models/classification/2502Feb04_18-57-08_classification_l0.0001/ckpts/model_ckpt_10_400.pth.tar',
        input_image_path='/mnt/disk1/data/generated_dataset/input_images/7041.png',
        output_image_path='/mnt/disk1/data/generated_dataset/output_images/7041.png',
        input_image_dir="",
        output_image_dir="",
        save_dir='',
        show=False,
        gpu=0
    )

    pred_score = main(args)

    print(f'{args.input_image_path.split("/")[-1]}: predicted score is {pred_score}')

