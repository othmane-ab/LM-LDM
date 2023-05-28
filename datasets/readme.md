# Image and Story Datasets
This Python module defines two torch.utils.data.Dataset subclasses: ImageDataset and StoryDataset. These classes are used for processing and loading image and text data for machine learning models.

## Dependencies
The module uses libraries including torch, torchvision, PIL, pandas, cv2, numpy, h5py, and transformers.

## Usage
### ImageDataset
- Initializes with subset and args parameters.
- subset defines the dataset split, i.e., 'train', 'val', or 'test'.
- args is a dictionary that holds arguments like the CSV file location, dataset name, and maximum caption length.
- Images are loaded from paths defined in the CSV file and processed using transforms.
- Text captions are tokenized using CLIPTokenizer.
### StoryDataset
- Like ImageDataset, it initializes with subset and args.
- Reads HDF5 files instead of CSVs. Each entry contains 5 images and corresponding text.
- The text and images are transformed and tokenized using CLIPTokenizer and a custom tokenizer (blip_tokenizer).
- If 'continuation' task is specified, it returns four images and their respective captions, alongside the original images and the source caption.

## Note
Both classes implement __len__() and __getitem__(self, index) methods, allowing PyTorch's DataLoader to