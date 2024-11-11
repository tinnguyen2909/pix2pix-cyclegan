from collections import namedtuple
import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from pathlib import Path

DSFile = namedtuple("DSFile", ["PathA", "PathB", "Filename", "ExtA", "ExtB"])


def get_common_file_names(pathA, pathB, extA, extB):
    # Get the list of file names (without extension) in each directory with the specified extension
    filesA = {file.stem for file in Path(pathA).glob(f'*{extA}')}
    filesB = {file.stem for file in Path(pathB).glob(f'*{extB}')}

    # Find the intersection of file names
    common_files = [DSFile(pathA, pathB, file, extA, extB)
                    for file in list(filesA.intersection(filesB))]

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
        # handle multiple paths
        # import pdb
        # pdb.set_trace()
        path_As = [p.strip() for p in opt.path_A.split(',')]
        path_Bs = [p.strip() for p in opt.path_B.split(',')]
        ext_As = [e.strip() for e in opt.ext_A.split(',')]
        ext_Bs = [e.strip() for e in opt.ext_B.split(',')]
        self.dataset_files = []
        assert len(path_As) == len(
            path_Bs), "path_A and path_B should have the same number of paths"
        assert len(ext_As) == len(ext_Bs)
        assert len(ext_As) == len(path_As)
        assert len(ext_Bs) == len(path_Bs)
        for path_A, path_B, ext_A, ext_B in zip(path_As, path_Bs, ext_As, ext_Bs):

            path_A = Path(path_A) / opt.phase
            path_B = Path(path_B) / opt.phase

            files = get_common_file_names(
                path_A, path_B, ext_A, ext_B)  # extension stripped (just name)

            self.dataset_files += files
        assert (self.opt.load_size >= self.opt.crop_size)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

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
        pathA, pathB, filename, extA, extB = self.dataset_files[index]
        A = Image.open(pathA / f"{filename}.{extA}")
        B = Image.open(pathB / f"{filename}.{extB}")
        # check if B has an alpha channel
        if B.mode == 'RGBA':
            B_alpha = B.split()[-1]  # alpha channel
            A_split = A.split()
            # replace B's alpha channel with A's alpha channel
            A = Image.merge(
                "RGBA", (A_split[0], A_split[1], A_split[2], B_alpha))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(
            self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(
            self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': str(pathA), 'B_paths': str(pathB)}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dataset_files)
