import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from pathlib import Path


def get_common_file_names(pathA, pathB, extA, extB):
    # Get the list of file names (without extension) in each directory with the specified extension
    filesA = {file.stem for file in Path(pathA).glob(f'*{extA}')}
    filesB = {file.stem for file in Path(pathB).glob(f'*{extB}')}

    # Find the intersection of file names
    common_files = list(filesA.intersection(filesB))

    return common_files


class PngAlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # get the image directory
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # self.AB_paths = sorted(make_dataset(
        #     self.dir_AB, opt.max_dataset_size))  # get image paths
        # crop_size should be smaller than the size of loaded image
        self.path_A = Path(opt.path_A) / opt.phase
        self.path_B = Path(opt.path_B) / opt.phase
        self.ext_A = opt.ext_A
        self.ext_B = opt.ext_B
        self.filenames = get_common_file_names(
            opt.path_A, opt.path_B, opt.ext_A, opt.ext_B)  # extension stripped (just name)
        assert (self.opt.load_size >= self.opt.crop_size)
        # self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # # read a image given a random integer index
        # AB_path = self.AB_paths[index]
        # # AB = Image.open(AB_path).convert('RGB')
        # AB = Image.open(AB_path)
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        A = Image.open(self.path_A / f"{self.filenames[index]}.{self.ext_A}")
        B = Image.open(self.path_B / f"{self.filenames[index]}.{self.ext_B}")
        B_alpha = B.split()[-1]  # alpha channel
        A_split = A.split()
        A = Image.merge("RGBA", (A_split[0], A_split[1], A_split[2], B_alpha))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(
            self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(
            self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': self.path_A, 'B_paths': self.path_B}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.filenames)
