from collections import namedtuple
import os
import random

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageEnhance
from pathlib import Path

DSFile = namedtuple(
    "DSFile", ["Path", "MaskPath", "Filename", "Ext", "MaskExt"])


def get_dataset_file_names(pathA, pathB, extA, extB):
    # Get the list of file names (without extension) in each directory with the specified extension
    filesA = {file.stem for file in Path(pathA).glob(f'*{extA}')}
    filesB = {file.stem for file in Path(pathB).glob(f'*{extB}')}

    # # Find the intersection of file names
    # common_files = [DSFile(pathA, pathB, file, extA, extB)
    #                 for file in list(filesA.intersection(filesB))]
    A_files = []
    B_files = []
    for file in filesA:
        if file in filesB:
            # for paired images, where image B has background removed, so we merge alpha channel of B into A
            A_files.append(DSFile(pathA, pathB, file, extA, extB))
        else:
            A_files.append(DSFile(pathA, None, file, extA, None))

    for file in filesB:
        B_files.append(DSFile(pathB, None, file, extB, None))
    return A_files, B_files


class MyUnalignedDataset(BaseDataset):
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
        self.A_files = []
        self.B_files = []
        assert len(ext_As) == len(ext_Bs)
        assert len(ext_As) == len(path_As)
        assert len(ext_Bs) == len(path_Bs)
        for path_A, path_B, ext_A, ext_B in zip(path_As, path_Bs, ext_As, ext_Bs):
            # flip = False
            # if path_A.endswith("__flip") and path_B.endswith("__flip"):
            #     flip = True
            #     path_A = path_A[:-6]
            #     path_B = path_B[:-6]
            path_A = Path(path_A) / opt.phase
            path_B = Path(path_B) / opt.phase

            A_files, B_files = get_dataset_file_names(
                path_A, path_B, ext_A, ext_B)  # extension stripped (just name)

            self.A_files.extend(A_files)
            self.B_files.extend(B_files)
        self.A_size = len(self.A_files)
        self.B_size = len(self.B_files)
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
        A_path, A_mask_path, A_filename, A_ext, A_mask_ext = self.A_files[index %
                                                                          self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path, _, B_filename, B_ext, _ = self.B_files[index_B]
        A = Image.open(A_path / f"{A_filename}.{A_ext}")
        B = Image.open(B_path / f"{B_filename}.{B_ext}")
        # check if B has an alpha channel
        if A_mask_path:
            A_mask = Image.open(A_mask_path / f"{A_filename}.{A_mask_ext}")
            if A_mask.mode == 'RGBA':
                A_mask = A_mask.split()[-1]  # alpha channel
                A_split = A.split()
                # replace B's alpha channel with A's alpha channel
                A = Image.merge(
                    "RGBA", (A_split[0], A_split[1], A_split[2], A_mask))
        if A.mode != "RGBA":
            A = A.convert("RGBA")
        if B.mode != "RGBA":
            B = B.convert("RGBA")
        if hasattr(self.opt, "adjust_brightness") and self.opt.adjust_brightness:
            A = ImageEnhance.Brightness(A).enhance(random.uniform(0.5, 1.5))
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(
            self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(
            self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': str(A_path), 'B_paths': str(B_path)}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(len(self.A_files), len(self.B_files))
