import os
import shutil
import zipfile
import numpy as np
import torch
import torch.utils.data as data
import requests
import tqdm
import tarfile
from PIL import Image
import pickle
import requests
import torchvision.transforms.functional as TF
CHUNK_SIZE = 1 * 1024 * 1024
from torch.utils.data import Dataset, ConcatDataset
import io
from torchvision.datasets import ImageFolder

def download_file(source, destination, size=None):
    if size is None:
        size = 0
    req = requests.get(source, stream=True)
    with open(destination, 'wb') as archive:
        for chunk in tqdm.tqdm(
            req.iter_content(chunk_size=CHUNK_SIZE),
            total=size // CHUNK_SIZE,
            leave=False,
        ):
            if chunk:
                archive.write(chunk)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

class CIFARFS(ImageFolder):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/cifarfs.py)

    **Description**

    The CIFAR Few-Shot dataset as originally introduced by Bertinetto et al., 2019.

    It consists of 60'000 colour images of sizes 32x32 pixels.
    The dataset is divided in 3 splits of 64 training, 16 validation, and 20 testing classes each containing 600 examples.
    The classes are sampled from the CIFAR-100 dataset, and we use the splits from Bertinetto et al., 2019.

    **References**

    1. Bertinetto et al. 2019. "Meta-learning with differentiable closed-form solvers". ICLR.

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.CIFARFS(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskGenerator(dataset=train_dataset, ways=ways)
    ~~~

    """

    def __init__(self,
                 root,
                 mode='train',
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.processed_root = os.path.join(self.root, 'cifarfs', 'processed')
        self.raw_path = os.path.join(self.root, 'cifarfs')

        if not self._check_exists() and download:
            self._download()
        if not self._check_processed():
            self._process_zip()
        mode = 'val' if mode == 'validation' else mode
        self.processed_root = os.path.join(self.processed_root, mode)
        self._bookkeeping_path = os.path.join(self.root, 'cifarfs-' + mode + '-bookkeeping.pkl')
        super(CIFARFS, self).__init__(root=self.processed_root,
                                      transform=self.transform,
                                      target_transform=self.target_transform)
        self.y = None
    def _check_exists(self):
        return os.path.exists(self.raw_path)

    def _check_processed(self):
        return os.path.exists(self.processed_root)

    def _download(self):
        # Download the zip, unzip it, and clean up
        print('Downloading CIFARFS to ', self.root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        zip_file = os.path.join(self.root, 'cifarfs.zip')
        try:
            download_file(
                source='https://zenodo.org/record/7978538/files/cifar100.zip',
                destination=zip_file,
            )
            with zipfile.ZipFile(zip_file, 'r') as zfile:
                zfile.extractall(self.raw_path)
            os.remove(zip_file)
        except Exception:
            download_file_from_google_drive('1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI',
                                            zip_file)
            with zipfile.ZipFile(zip_file, 'r') as zfile:
                zfile.extractall(self.raw_path)
            os.remove(zip_file)

    def _process_zip(self):
        print('Creating CIFARFS splits')
        if not os.path.exists(self.processed_root):
            os.mkdir(self.processed_root)
        split_path = os.path.join(self.raw_path, 'cifar100', 'splits', 'bertinetto')
        train_split_file = os.path.join(split_path, 'train.txt')
        valid_split_file = os.path.join(split_path, 'val.txt')
        test_split_file = os.path.join(split_path, 'test.txt')

        source_dir = os.path.join(self.raw_path, 'cifar100', 'data')
        for fname, dest in [(train_split_file, 'train'),
                            (valid_split_file, 'val'),
                            (test_split_file, 'test')]:
            dest_target = os.path.join(self.processed_root, dest)
            if not os.path.exists(dest_target):
                os.mkdir(dest_target)
            with open(fname) as split:
                for label in split.readlines():
                    source = os.path.join(source_dir, label.strip())
                    target = os.path.join(dest_target, label.strip())
                    shutil.copytree(source, target)

DATA_DIR = 'cubirds200'
DATA_FILENAME = 'CUB_200_2011.tgz'
ARCHIVE_ID = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
ZENODO_URL = 'https://zenodo.org/record/8000562/files/CUB_200_2011.tgz'

SPLITS = {
    'train': [
        '190.Red_cockaded_Woodpecker',
        '144.Common_Tern',
        '014.Indigo_Bunting',
        '012.Yellow_headed_Blackbird',
        '059.California_Gull',
        '031.Black_billed_Cuckoo',
        '071.Long_tailed_Jaeger',
        '018.Spotted_Catbird',
        '177.Prothonotary_Warbler',
        '040.Olive_sided_Flycatcher',
        '063.Ivory_Gull',
        '073.Blue_Jay',
        '166.Golden_winged_Warbler',
        '160.Black_throated_Blue_Warbler',
        '016.Painted_Bunting',
        '149.Brown_Thrasher',
        '126.Nelson_Sharp_tailed_Sparrow',
        '090.Red_breasted_Merganser',
        '074.Florida_Jay',
        '058.Pigeon_Guillemot',
        '105.Whip_poor_Will',
        '043.Yellow_bellied_Flycatcher',
        '158.Bay_breasted_Warbler',
        '192.Downy_Woodpecker',
        '129.Song_Sparrow',
        '161.Blue_winged_Warbler',
        '132.White_crowned_Sparrow',
        '146.Forsters_Tern',
        '011.Rusty_Blackbird',
        '070.Green_Violetear',
        '197.Marsh_Wren',
        '041.Scissor_tailed_Flycatcher',
        '100.Brown_Pelican',
        '120.Fox_Sparrow',
        '032.Mangrove_Cuckoo',
        '119.Field_Sparrow',
        '183.Northern_Waterthrush',
        '007.Parakeet_Auklet',
        '053.Western_Grebe',
        '001.Black_footed_Albatross',
        '102.Western_Wood_Pewee',
        '164.Cerulean_Warbler',
        '036.Northern_Flicker',
        '131.Vesper_Sparrow',
        '098.Scott_Oriole',
        '188.Pileated_Woodpecker',
        '139.Scarlet_Tanager',
        '107.Common_Raven',
        '108.White_necked_Raven',
        '184.Louisiana_Waterthrush',
        '099.Ovenbird',
        '171.Myrtle_Warbler',
        '075.Green_Jay',
        '097.Orchard_Oriole',
        '152.Blue_headed_Vireo',
        '173.Orange_crowned_Warbler',
        '095.Baltimore_Oriole',
        '042.Vermilion_Flycatcher',
        '054.Blue_Grosbeak',
        '079.Belted_Kingfisher',
        '006.Least_Auklet',
        '142.Black_Tern',
        '078.Gray_Kingbird',
        '047.American_Goldfinch',
        '050.Eared_Grebe',
        '037.Acadian_Flycatcher',
        '196.House_Wren',
        '083.White_breasted_Kingfisher',
        '062.Herring_Gull',
        '138.Tree_Swallow',
        '060.Glaucous_winged_Gull',
        '182.Yellow_Warbler',
        '027.Shiny_Cowbird',
        '174.Palm_Warbler',
        '157.Yellow_throated_Vireo',
        '117.Clay_colored_Sparrow',
        '175.Pine_Warbler',
        '024.Red_faced_Cormorant',
        '106.Horned_Puffin',
        '151.Black_capped_Vireo',
        '005.Crested_Auklet',
        '185.Bohemian_Waxwing',
        '049.Boat_tailed_Grackle',
        '010.Red_winged_Blackbird',
        '153.Philadelphia_Vireo',
        '017.Cardinal',
        '023.Brandt_Cormorant',
        '115.Brewer_Sparrow',
        '104.American_Pipit',
        '109.American_Redstart',
        '167.Hooded_Warbler',
        '123.Henslow_Sparrow',
        '019.Gray_Catbird',
        '067.Anna_Hummingbird',
        '081.Pied_Kingfisher',
        '077.Tropical_Kingbird',
        '088.Western_Meadowlark',
        '048.European_Goldfinch',
        '141.Artic_Tern',
        '013.Bobolink',
        '029.American_Crow',
        '025.Pelagic_Cormorant',
        '135.Bank_Swallow',
        '056.Pine_Grosbeak',
        '179.Tennessee_Warbler',
        '087.Mallard',
        '195.Carolina_Wren',
        '038.Great_Crested_Flycatcher',
        '092.Nighthawk',
        '187.American_Three_toed_Woodpecker',
        '003.Sooty_Albatross',
        '004.Groove_billed_Ani',
        '156.White_eyed_Vireo',
        '180.Wilson_Warbler',
        '034.Gray_crowned_Rosy_Finch',
        '093.Clark_Nutcracker',
        '110.Geococcyx',
        '154.Red_eyed_Vireo',
        '143.Caspian_Tern',
        '089.Hooded_Merganser',
        '186.Cedar_Waxwing',
        '069.Rufous_Hummingbird',
        '125.Lincoln_Sparrow',
        '026.Bronzed_Cowbird',
        '111.Loggerhead_Shrike',
        '022.Chuck_will_Widow',
        '165.Chestnut_sided_Warbler',
        '021.Eastern_Towhee',
        '191.Red_headed_Woodpecker',
        '086.Pacific_Loon',
        '124.Le_Conte_Sparrow',
        '002.Laysan_Albatross',
        '033.Yellow_billed_Cuckoo',
        '189.Red_bellied_Woodpecker',
        '116.Chipping_Sparrow',
        '130.Tree_Sparrow',
        '114.Black_throated_Sparrow',
        '065.Slaty_backed_Gull',
        '091.Mockingbird',
        '181.Worm_eating_Warbler',
    ],
    'test': [
        '008.Rhinoceros_Auklet',
        '009.Brewer_Blackbird',
        '015.Lazuli_Bunting',
        '020.Yellow_breasted_Chat',
        '028.Brown_Creeper',
        '030.Fish_Crow',
        '035.Purple_Finch',
        '039.Least_Flycatcher',
        '045.Northern_Fulmar',
        '046.Gadwall',
        '082.Ringed_Kingfisher',
        '085.Horned_Lark',
        '094.White_breasted_Nuthatch',
        '101.White_Pelican',
        '103.Sayornis',
        '112.Great_Grey_Shrike',
        '118.House_Sparrow',
        '122.Harris_Sparrow',
        '128.Seaside_Sparrow',
        '133.White_throated_Sparrow',
        '134.Cape_Glossy_Starling',
        '137.Cliff_Swallow',
        '147.Least_Tern',
        '148.Green_tailed_Towhee',
        '163.Cape_May_Warbler',
        '168.Kentucky_Warbler',
        '169.Magnolia_Warbler',
        '170.Mourning_Warbler',
        '193.Bewick_Wren',
        '194.Cactus_Wren',
    ],
    'validation': [
        '044.Frigatebird',
        '051.Horned_Grebe',
        '052.Pied_billed_Grebe',
        '055.Evening_Grosbeak',
        '057.Rose_breasted_Grosbeak',
        '061.Heermann_Gull',
        '064.Ring_billed_Gull',
        '066.Western_Gull',
        '068.Ruby_throated_Hummingbird',
        '072.Pomarine_Jaeger',
        '076.Dark_eyed_Junco',
        '080.Green_Kingfisher',
        '084.Red_legged_Kittiwake',
        '096.Hooded_Oriole',
        '113.Baird_Sparrow',
        '121.Grasshopper_Sparrow',
        '127.Savannah_Sparrow',
        '136.Barn_Swallow',
        '140.Summer_Tanager',
        '145.Elegant_Tern',
        '150.Sage_Thrasher',
        '155.Warbling_Vireo',
        '159.Black_and_white_Warbler',
        '162.Canada_Warbler',
        '172.Nashville_Warbler',
        '176.Prairie_Warbler',
        '178.Swainson_Warbler',
        '198.Rock_Wren',
        '199.Winter_Wren',
        '200.Common_Yellowthroat',
    ]
}

IMAGENET_DUPLICATES = {
    'train': [
        'American_Goldfinch_0062_31921.jpg',
        'Indigo_Bunting_0063_11820.jpg',
        'Blue_Jay_0053_62744.jpg',
        'American_Goldfinch_0131_32911.jpg',
        'Indigo_Bunting_0051_12837.jpg',
        'American_Goldfinch_0012_32338.jpg',
        'Laysan_Albatross_0033_658.jpg',
        'Black_Footed_Albatross_0024_796089.jpg',
        'Indigo_Bunting_0072_14197.jpg',
        'Green_Violetear_0002_795699.jpg',
        'Black_Footed_Albatross_0033_796086.jpg',
        'Black_Footed_Albatross_0086_796062.jpg',
        'Anna_Hummingbird_0034_56614.jpg',
        'American_Goldfinch_0064_32142.jpg',
        'Red_Breasted_Merganser_0068_79203.jpg',
        'Blue_Jay_0033_62024.jpg',
        'Indigo_Bunting_0071_11639.jpg',
        'Red_Breasted_Merganser_0001_79199.jpg',
        'Indigo_Bunting_0060_14495.jpg',
        'Laysan_Albatross_0053_543.jpg',
        'American_Goldfinch_0018_32324.jpg',
        'Red_Breasted_Merganser_0034_79292.jpg',
        'Mallard_0067_77623.jpg',
        'Red_Breasted_Merganser_0083_79562.jpg',
        'Laysan_Albatross_0049_918.jpg',
        'Black_Footed_Albatross_0002_55.jpg',
        'Red_Breasted_Merganser_0012_79425.jpg',
        'Indigo_Bunting_0031_13300.jpg',
        'Blue_Jay_0049_63082.jpg',
        'Indigo_Bunting_0010_13000.jpg',
        'Red_Breasted_Merganser_0004_79232.jpg',
        'Red_Breasted_Merganser_0045_79358.jpg',
        'American_Goldfinch_0116_31943.jpg',
        'Blue_Jay_0068_61543.jpg',
        'Indigo_Bunting_0073_13933.jpg',
    ],
    'validation': [
        'Dark_Eyed_Junco_0057_68650.jpg',
        'Dark_Eyed_Junco_0102_67402.jpg',
        'Ruby_Throated_Hummingbird_0090_57411.jpg',
        'Dark_Eyed_Junco_0031_66785.jpg',
        'Dark_Eyed_Junco_0037_66321.jpg',
        'Dark_Eyed_Junco_0111_66488.jpg',
        'Ruby_Throated_Hummingbird_0040_57982.jpg',
        'Dark_Eyed_Junco_0104_67820.jpg',
    ],
    'test': [],
}
IMAGENET_DUPLICATES['all'] = sum(IMAGENET_DUPLICATES.values(), [])


class CUBirds200(torch.utils.data.Dataset):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/cu_birds200.py)

    **Description**

    The Caltech-UCSD Birds dataset was originally introduced by Wellinder et al., 2010 and then re-purposed for few-shot learning in Triantafillou et al., 2020.

    The dataset consists of 6,033 bird images classified into 200 bird species.
    The train set consists of 140 classes, while the validation and test sets each contain 30.
    We provide the raw (unprocessed) images, and follow the train-validation-test splits of Triantafillou et al.

    This dataset includes 43 images that overlap with the ILSVRC-2012 (ImageNet) dataset.
    They are omitted by default, but can be included by setting the `include_imagenet_duplicates` flag to `True`.

    **References**

    1. Welinder et al. 2010. "Caltech-UCSD Birds 200." Caltech Technical Report.
    2. Triantafillou et al. 2020. "Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples." ICLR '20.
    3. [http://www.vision.caltech.edu/visipedia/CUB-200.html](http://www.vision.caltech.edu/visipedia/CUB-200.html)

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.
    * **include_imagenet_duplicates** (bool, *optional*, default=False) - Whether to include images that are also present in the ImageNet 2012 dataset.
    * **bounding_box_crop** (bool, *optional*, default=False) - Whether to crop each image using bounding box information.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.CUBirds200(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.Taskset(dataset=train_dataset, num_tasks=1000)
    ~~~

    """

    def __init__(
        self,
        root,
        mode='all',
        transform=None,
        target_transform=None,
        download=False,
        include_imagenet_duplicates=False,
        bounding_box_crop=False,
    ):
        root = os.path.expanduser(root)
        self.root = root
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.include_imagenet_duplicates = include_imagenet_duplicates
        self.bounding_box_crop = bounding_box_crop
        self._bookkeeping_path = os.path.join(
            self.root,
            'cubirds200-' + mode + '-bookkeeping.pkl'
        )

        if not self._check_exists() and download:
            self.download()

        self.load_data(mode)

    def _check_exists(self):
        data_path = os.path.join(self.root, DATA_DIR)
        return os.path.exists(data_path)

    def download(self):
        # Download and extract the data
        data_path = os.path.join(self.root, DATA_DIR)
        os.makedirs(data_path, exist_ok=True)
        tar_path = os.path.join(data_path, DATA_FILENAME)
        print('Downloading CUBirds200 dataset. (1.1Gb)')
        try:
            download_file(ZENODO_URL, tar_path)
            tar_file = tarfile.open(tar_path)
            tar_file.extractall(data_path)
            tar_file.close()
            os.remove(tar_path)
        except Exception:
            download_file_from_google_drive(ARCHIVE_ID, tar_path)
            tar_file = tarfile.open(tar_path)
            tar_file.extractall(data_path)
            tar_file.close()
            os.remove(tar_path)

    def load_data(self, mode='train'):
        classes = sum(SPLITS.values(), []) if mode == 'all' else SPLITS[mode]
        images_path = os.path.join(
            self.root,
            DATA_DIR,
            'CUB_200_2011',
            'images',
        )
        duplicates = IMAGENET_DUPLICATES[self.mode]
        self.data = []

        # parse bounding boxes
        if self.bounding_box_crop:
            self.bounding_boxes = {}
            bbox_file = os.path.join(self.root, DATA_DIR, 'CUB_200_2011', 'bounding_boxes.txt')
            id2img_file = os.path.join(self.root, DATA_DIR, 'CUB_200_2011', 'images.txt')
            with open(bbox_file, 'r') as bbox_fd:
                content = bbox_fd.readlines()
            id2img = {}
            with open(id2img_file, 'r') as id2img_fd:
                for line in id2img_fd.readlines():
                    line = line.replace('\n', '').split(' ')
                    id2img[line[0]] = line[1]
            bbox_content = {}
            for line in content:
                line = line.split(' ')
                x, y, width, height = (
                    int(float(line[1])),
                    int(float(line[2])),
                    int(float(line[3])),
                    int(float(line[4])),
                )
                bbox_content[id2img[line[0]]] = (x, y, x+width, y+height)

        # read images from disk
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(images_path, class_name)
            filenames = os.listdir(class_path)
            for image_file in filenames:
                if self.include_imagenet_duplicates or \
                   image_file not in duplicates:
                    image_path = os.path.join(class_path, image_file)
                    if self.bounding_box_crop:
                        self.bounding_boxes[image_path] = bbox_content[os.path.join(class_name, image_file)]
                    self.data.append((image_path, class_idx))

    def __getitem__(self, i):
        image_path, label = self.data[i]
        image = Image.open(image_path).convert('RGB')
        if self.bounding_box_crop:
            bbox = self.bounding_boxes[image_path]
            image = image.crop(bbox)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        length = len(self.data)
        return length
def index_classes(items):
    idx = {}
    for i in items:
        if (i not in idx):
            idx[i] = len(idx)
    return idx

class TieredImagenet(data.Dataset):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/tiered_imagenet.py)

    **Description**

    The *tiered*-ImageNet dataset was originally introduced by Ren et al, 2018 and we download the data directly from the link provided in their repository.

    Like *mini*-ImageNet, *tiered*-ImageNet builds on top of ILSVRC-12, but consists of 608 classes (779,165 images) instead of 100.
    The train-validation-test split is made such that classes from similar categories are in the same splits.
    There are 34 categories each containing between 10 and 30 classes.
    Of these categories, 20 (351 classes; 448,695 images) are used for training,
    6 (97 classes; 124,261 images) for validation, and 8 (160 class; 206,209 images) for testing.

    **References**

    1. Ren et al, 2018. "Meta-Learning for Semi-Supervised Few-Shot Classification." ICLR '18.
    2. Ren Mengye. 2018. "few-shot-ssl-public". [https://github.com/renmengye/few-shot-ssl-public](https://github.com/renmengye/few-shot-ssl-public)

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.TieredImagenet(root='./data', mode='train', download=True)
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.Taskset(dataset=train_dataset, num_tasks=1000)
    ~~~

    """

    def __init__(self, root, mode='train', transform=None, target_transform=None, download=False):
        super(TieredImagenet, self).__init__()
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.transform = transform
        self.target_transform = target_transform
        if mode not in ['train', 'validation', 'test']:
            raise ValueError('mode must be train, validation, or test.')
        self.mode = mode
        self._bookkeeping_path = os.path.join(self.root, 'tiered-imagenet-bookkeeping-' + mode + '.pkl')
        google_drive_file_id = '1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07'

        if not self._check_exists() and download:
            self.download(google_drive_file_id, self.root)

        short_mode = 'val' if mode == 'validation' else mode
        tiered_imaganet_path = os.path.join(self.root, 'tiered-imagenet')
        images_path = os.path.join(tiered_imaganet_path, short_mode + '_images_png.pkl')
        with open(images_path, 'rb') as images_file:
            self.images = pickle.load(images_file)
        labels_path = os.path.join(tiered_imaganet_path, short_mode + '_labels.pkl')
        with open(labels_path, 'rb') as labels_file:
            self.labels = pickle.load(labels_file)
            self.labels = self.labels['label_specific']

    def download(self, file_id, destination):
        print('Downloading tiered ImageNet. (12Gb) Please be patient.')
        try:
            archive_dir = os.path.join(destination, 'tiered-imagenet')
            os.makedirs(archive_dir, exist_ok=True)
            files_to_download = [
                'https://zenodo.org/record/7978538/files/tiered-imagenet-class_names.txt',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-synsets.txt',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-test_images_png.pkl',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-test_labels.pkl',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-train_images_png.pkl',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-train_labels.pkl',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-val_images_png.pkl',
                'https://zenodo.org/record/7978538/files/tiered-imagenet-val_labels.pkl',
            ]
            for file_url in files_to_download:
                file_dest = os.path.join(
                    archive_dir,
                    os.path.basename(file_url).replace('tiered-imagenet-', '')
                )
                download_file(
                    source=file_url,
                    destination=file_dest,
                )
        except Exception:
            archive_path = os.path.join(destination, 'tiered_imagenet.tar')
            download_file_from_google_drive(file_id, archive_path)
            archive_file = tarfile.open(archive_path)
            archive_file.extractall(destination)
            os.remove(archive_path)

    def __getitem__(self, idx):
        image = Image.open(io.BytesIO(self.images[idx]))
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.labels)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           'tiered-imagenet',
                                           'train_images_png.pkl'))
