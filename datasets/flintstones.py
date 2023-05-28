import random

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

# from models.blip_override.blip import init_tokenizer


import torch
from PIL import Image
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, subset, args):
        super(ImageDataset, self).__init__()

        self.args = args
        self.csv_file = args.get(args.dataset).csv_file

        self.data = pd.read_csv(self.csv_file)
        self.data = (self.data[:int(self.data.shape[0]*0.8)] if subset=="train" else self.data[int(self.data.shape[0]*0.8):int(self.data.shape[0]*0.9)] if subset=="val" else self.data[int(self.data.shape[0]*0.9):]).reset_index()
        self.image_paths = "/".join(self.csv_file.split("/")[:-1]) + "/" + self.data['image_path']
        self.captions = self.data['caption']
        
        self.augment = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.subset = subset
        self.dataset = args.dataset

        self.max_length = args.get(args.dataset).max_length

        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        msg = self.clip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens))
        print("clip {} new tokens added".format(msg))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        caption = self.captions[index]
        
        # Load image
        image = Image.open(image_path)
        image = self.augment(image) if self.subset in ['train', 'val'] else torch.from_numpy(np.array(image))
        
               
        text = self.data['caption'][index]

        # tokenize caption using default tokenizer
        tokenized = self.clip_tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        caption, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

        return image, caption, attention_mask

class StoryDataset(Dataset):
    """
    A custom subset class for the LRW (includes train, val, test) subset
    """

    def __init__(self, subset, args):
        super(StoryDataset, self).__init__()
        self.args = args

        self.h5_file = args.get(args.dataset).hdf5_file
        self.subset = subset

        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.dataset = args.dataset
        self.max_length = args.get(args.dataset).max_length
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()
        msg = self.clip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens))
        print("clip {} new tokens added".format(msg))
        msg = self.blip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens))
        print("blip {} new tokens added".format(msg))

        self.blip_image_processor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),# scales to 0,1
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]) ## (pixel[r,g,b]-mean)/std
        ])

    def open_h5(self):
        h5 = h5py.File(self.h5_file, "r")
        self.h5 = h5[self.subset]

    def __getitem__(self, index):
        if not hasattr(self, 'h5'):
            self.open_h5()

        images = list()
        for i in range(5):
            im = self.h5['image{}'.format(i)][index]
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
            idx = random.randint(0, 4)
            images.append(im[idx * 128: (idx + 1) * 128])#selects a random portion of the image vertically with a height of 128 pixels

        source_images = torch.stack([self.blip_image_processor(im) for im in images])
        images = images[1:] if self.args.task == 'continuation' else images
        images = torch.stack([self.augment(im) for im in images]) \
            if self.subset in ['train', 'val'] else torch.from_numpy(np.array(images))

        texts = self.h5['text'][index].decode('utf-8').split('|')

        # tokenize caption using default tokenizer
        tokenized = self.clip_tokenizer(
            texts[1:] if self.args.task == 'continuation' else texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        captions, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

        tokenized = self.blip_tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        source_caption, source_attention_mask = tokenized['input_ids'], tokenized['attention_mask']
        return images, captions, attention_mask, source_images, source_caption, source_attention_mask

    def __len__(self):
        if not hasattr(self, 'h5'):
            self.open_h5()
        return len(self.h5['text'])


# if __name__=="__main__":
#     from torch.utils.data import DataLoader
#     # Define your arguments
#     class Args:
#         dataset = 'flintstones'
#         flintstones = {
#             "csv_file" : '/home/nlab/abouelaecha/finetune_stable_diffusion/data/data.csv',
#             "max_length" : 91,
#             "new_tokens" : [ "fred", "barney", "wilma", "betty", "pebbles", "dino", "slate" ],
#         }
#         def get(self, _dataset):
#             return getattr(self, _dataset)

#     args = Args()

#     # Create an instance of the ImageDataset
#     dataset = ImageDataset(subset='train', args=args)

#     # Example usage
#     index = 0
#     image, caption, attention_mask = dataset[index]

#     # Print the shapes
#     print("Image shape:", image.shape)
#     print("Caption shape:", caption.shape)
#     print("Attention mask shape:", attention_mask.shape)

#     # Create a data loader
#     batch_size = 16
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # Iterate over the data loader
#     for images, captions, attention_masks in dataloader:
#         # Perform operations on the batched data
#         # For example, pass the images and captions through your model
#         # ...
#         # ...
#         # Perform any necessary calculations or computations
#         # ...

#         # Print the shapes of the batched data
#         print("Batched images shape:", images.shape)
#         print("Batched captions shape:", captions.shape)
#         print("Batched attention masks shape:", attention_masks.shape)
