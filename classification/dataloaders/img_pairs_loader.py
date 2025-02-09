import os
import numpy as np
import itertools
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from dataloaders import img_transforms as transforms
from dataloaders.dataloader_builder import DataLoader


class img_pairs_loader(DataLoader):
    def __init__(self, FLAGS, mode):
        super(img_pairs_loader, self).__init__(FLAGS, mode)
        self.data_dir = FLAGS.data_dir
        self.mode = mode
        self.batch_size = FLAGS.batch_size
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.num_channels = FLAGS.num_channels
        self.num_classes = FLAGS.num_classes
        self.crop_shift = 1
        if hasattr(FLAGS, 'crop_shift'):
            self.crop_shift = FLAGS.crop_shift
        if hasattr(FLAGS, 'shuffle'):
            self.shuffle = FLAGS.shuffle
        else:
            if mode != 'train':
                self.shuffle = False
            else:
                self.shuffle = True
        self.hflip = None
        if hasattr(FLAGS, 'hflip'):
            self.hflip = FLAGS.hflip
        self.max_rot_angle = 0
        if hasattr(FLAGS, 'max_rot_angle'):
            self.max_rot_angle = FLAGS.max_rot_angle
        self.color = 1
        if hasattr(FLAGS, 'color'):
            self.color = FLAGS.color
        self.contrast = 1
        if hasattr(FLAGS, 'contrast'):
            self.contrast = FLAGS.contrast
        self.brightness = 1
        if hasattr(FLAGS, 'brightness'):
            self.brightness = FLAGS.brightness
        self.sharpness = 1
        if hasattr(FLAGS, 'sharpness'):
            self.sharpness = FLAGS.sharpness
        self.noise = False
        if hasattr(FLAGS, 'noise'):
            self.noise = FLAGS.noise

        self.transforms = transforms.Compose([
            transforms.RandHFlip(self.hflip),
            transforms.Rotate(self.max_rot_angle),
            transforms.RandCrop((self.crop_shift, self.crop_shift)),
            transforms.EnhanceColor(self.color),
            transforms.EnhanceContrast(self.contrast),
            transforms.EnhanceBrightness(self.brightness),
            transforms.EnhanceSharpness(self.sharpness),
            transforms.RandNoise(self.noise),
            transforms.ResizeAndFillBlack((self.width, self.height)),
        ])

        self.input_images_dir = os.path.join(self.subset_datadir, 'input_images')
        self.output_images_dir = os.path.join(self.subset_datadir, 'output_images')
        self.input_images_list, self.output_images_list, self.num_samples = self._filelist()
        self.picture_names, self.scores = self._load_annotations()
        self._remove_imgs_without_scores()
        self.num_batches = self.num_samples // self.batch_size


    def build(self):
        num_outputs = 4
        gen = self.batch_generator(self._image_generator(), num_outputs=num_outputs)
        return gen, self.num_samples


    def len(self):
        return self.num_samples


    def _filelist(self):
        input_images_list = sorted(os.listdir(self.input_images_dir))
        input_images_list = [name for name in input_images_list if name.endswith('.png')]
        output_images_list = sorted(os.listdir(self.output_images_dir))
        output_images_list = [name for name in output_images_list if name.endswith('.png')]
        assert len(input_images_list) == len(output_images_list)
        self.num_samples = len(input_images_list)
        return input_images_list, output_images_list, self.num_samples


    def _load_annotations(self):
        csv_path = os.path.join(os.path.dirname(self.subset_datadir), "Scoring_annotation.csv")
        df = pd.read_csv(csv_path)
        picture_names = df["picture name"].tolist()
        scores = df["score"].tolist()
        picture_names_no_ext = [os.path.splitext(name)[0] for name in picture_names]
        return picture_names_no_ext, scores

    def _remove_imgs_without_scores(self):
        '''Removes images from the dataset that are not in the scoring annotation file'''
        for input_img, output_img in zip(self.input_images_list, self.output_images_list):
            input_img_no_ext = os.path.splitext(input_img)[0]
            output_img_no_ext = os.path.splitext(output_img)[0]
            assert input_img_no_ext == output_img_no_ext, \
                f"Input and output filenames do not match: {input_img} vs {output_img}"

            if not input_img_no_ext in self.picture_names:
                self.input_images_list.remove(input_img)
                self.output_images_list.remove(output_img)
        self.num_samples = len(self.input_images_list)

    def _idx_generator(self):
        '''Before each epoch shuffles image order.
        Each train/test set is given in a flat directory.
        Yields next index of an image'''
        # infinite loop over epochs
        for epoch in itertools.count():
            if self.shuffle:
                file_set_idxs = np.random.permutation(self.num_samples)
            else:
                file_set_idxs = range(self.num_samples)

            # loop over all training set (one epoch)
            for idx in file_set_idxs:
                yield idx


    def _image_generator(self):
        """
        Generate transformed images and their associated score (label).
        This generator does the following:
        1. Iterates over image indices (one epoch) using self._idx_generator().
        2. For each index, retrieves matching input and output image names.
        3. Opens both images, applies self.transforms to produce two transformed images
           (each of shape (Img_H, Img_W, n_channels)).
        4. Looks up the score for the corresponding picture name (without file extension).
        5. Yields a 4-tuple: (input_image, output_image, score, idx) where:
           - input_image: np.array of shape (Img_H, Img_W, n_channels)
           - output_image: np.array of shape (Img_H, Img_W, n_channels)
           - score: numeric score associated with the image pair
           - idx: index from the dataset iteration
        Raises:
            ValueError: If the filename (without extension) is not found in self.picture_names.
        """
        # Loop over all indices (one epoch)
        for idx in self._idx_generator():
            # Retrieve the input and output image names by index
            input_image_name = self.input_images_list[idx]
            output_image_name = self.output_images_list[idx]
            input_image_name_no_ext = os.path.splitext(input_image_name)[0]
            output_image_name_no_ext = os.path.splitext(output_image_name)[0]
            assert input_image_name_no_ext == output_image_name_no_ext, \
                f"Input and output filenames do not match: {input_image_name} vs {output_image_name}"

            # Look up the score for the current filename (without extension)
            if input_image_name_no_ext in self.picture_names:
                # Load images from disk
                input_im = Image.open(os.path.join(self.input_images_dir, input_image_name)).convert('RGB')
                output_im = Image.open(os.path.join(self.output_images_dir, output_image_name)).convert('RGB')

                # Apply transformations (returns something like (transformed_input, transformed_output))
                images = self.transforms(input_im, output_im)
                images_np = np.array(images[:2]).astype(np.uint8)
                input_image, output_image = images_np[0], images_np[1]

                # Find the corresponding index in picture_names
                score_idx = self.picture_names.index(input_image_name_no_ext)
                # Retrieve the score
                score = self.scores[score_idx]

                # Yield the input image, output image, score, and the current dataset index
                yield input_image, output_image, score, idx
            else:
                # Raise an error if we cannot find the filename in self.picture_names
                raise ValueError(f"Filename '{input_image_name_no_ext}' not found in picture_names.")
                yield None, None, None, None








if __name__ == '__main__':
    from dataloaders.dataloader_builder import DataLoader

    class FLAGS():
        def __init__(self):
            self.batch_size = 1
            self.data_dir = '/mnt/disk1/data/generated_dataset'
            self.height = 768
            self.width = 512
            self.data_loader = "img_pairs_loader"
            self.model = "ResNet"
            self.metric = "accuracy_categ"
            self.num_channels = 3
            self.num_classes = 4
            self.n_filters = "16,32"
            self.batch_norm = 1
            self.shuffle = 0
            self.hflip = 1
            self.crop_shift = 0.85
            self.max_rot_angle = 10
            self.color = 0.2
            self.contrast = 0.2
            self.brightness = 0.2
            self.sharpness = 0.2
            self.noise = 1

    FLAGS = FLAGS()
    dataloader, num_samples = DataLoader.dataloader_builder(FLAGS.data_loader, FLAGS, 'train')
    for i, (input_image, output_image, score, _) in enumerate(dataloader):
        print('\n\ninput_image: {}, \noutput_image{}\nscore: {}\n\n'.
              format(input_image.shape, output_image.shape, score[0]))


        # plt.imshow(Image.fromarray(input_image[0]))
        # plt.imshow(Image.fromarray(output_image[0]))
        # plt.axis('off')  # Turn off axes for better view
        # plt.show()


        concatenated_image = np.concatenate((input_image[0], output_image[0]), axis=1)
        plt.imshow(concatenated_image)
        plt.axis('off')  # Hide axes
        plt.show()
        print('done')













    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    