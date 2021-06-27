import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SatHeiDataset(BaseDataset):
    
    def __init__(self, opt):
        
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.paths = sorted(make_dataset(self.dir, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc =  self.opt.input_nc
        self.output_nc =  self.opt.output_nc

    def __getitem__(self, index):

        # read a image given a random integer index
        path = self.paths[index]
        ABC = Image.open(path).convert('RGB')
        # split AB image into A and B
        w, h = ABC.size
        w2 = int(w / 3)
        A = ABC.crop((0, 0, w2, h))
        B = ABC.crop((w2, 0, w2*2, h))
        C = ABC.crop((w2*2, 0, w2*3, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=1)
        B_transform = get_transform(self.opt, transform_params, grayscale=0)
        C_transform = get_transform(self.opt, transform_params, grayscale=1)

        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)

        return {'A': A, 'B': B,'C':C, 'A_paths': path, 'B_paths': path, 'C_paths': path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)
