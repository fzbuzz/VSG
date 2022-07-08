import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
from SceneGraphLanguage import SceneGraphLanguage
from torch.nn.utils.rnn import pad_sequence



class VRGym_with_objects(Dataset):
    ''' 
    VRGym as defined in README. 
    Dataset must be augmented with preprocess_vocab.py in scripts/
    '''
    def __init__(self, split='train', data="../data/basic_data_v4/", num_objects=10, size=(128,128), debug=False, frames=1):
        super(VRGym_with_objects, self).__init__()
        
        assert split in ['train', 'test']
        self.split = split
        self.root_dir = os.path.join(data, split)
        self.folders = list(filter(lambda x: x.isnumeric(), os.listdir(self.root_dir)))
        self.modalities = ['lit','depth','segment','objects']
        self.num_objects = num_objects
        self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size)])
        self.frames = frames
        self.l_embed = nn.Embedding.from_pretrained(self.get_embeddings(), freeze=True)
        self.debug = debug
        self.all_files = self.collect_vrgym_files()

        with open(os.path.join(self.root_dir,'w_to_i.json')) as f:
            self.w_to_i = json.load(f)
            self.i_to_w = {v: k for k, v in self.w_to_i.items()}
            self.i_to_w[len(self.i_to_w.keys())] = ''
        
    def __getitem__(self, index):
        data = self.all_files[index]
        
        if self.frames == 1: data = [data]
        frame_datas = []
        for data_obj in list(data):
            lit, depth, segment, objects = data_obj
            image = Image.open(lit).convert("RGB")
            segment = Image.open(segment).convert("RGB")

            with open(objects, 'r') as f:
                objects = json.load(f)
                np.random.shuffle(objects)
            '''
            truncates objects with max num object of num_objects. 
            pads if necessary with "empty" value
            creates mask with last value as True to indicate thats our bg slot
            '''
            objects_idx = list(map(lambda x: self.w_to_i[x], objects))[:self.num_objects]
            objects_idx = objects_idx + [len(self.w_to_i)]*(self.num_objects - len(objects_idx))
            padded_objects_idx = torch.LongTensor(objects_idx + [len(self.w_to_i)]) #add bg slot

            mask = [obj == len(self.w_to_i) for obj in objects_idx]
            mask.append(False)
            mask = torch.tensor(mask)
            
            image = self.img_transform(image)
            segment = self.img_transform(segment)

            if self.frames == 1:
                if self.debug: return {'image': image, 'segment': segment, 'objects': self.l_embed(padded_objects_idx), 'object_names': padded_objects_idx, 'mask': mask }
                return {'image': image, 'segment': segment, 'objects': self.l_embed(padded_objects_idx), 'mask': mask }

            frame_datas.append({'image': image, 'segment': segment, 'objects': self.l_embed(padded_objects_idx), 'object_names': padded_objects_idx })

        return frame_datas

    def get_embeddings(self):
        embeddings = torch.load(os.path.join(self.root_dir, 'all_embeddings.pt'))
        #return embeddings
        return torch.cat([embeddings, torch.zeros(1024).unsqueeze(0)])

    def index_to_word(self, idx):
        return list(map(lambda x: self.i_to_w[x], idx))

    def collect_vrgym_files(self):
        files = []
        for folder in self.folders:
            all_modalities = {}
            for modality in self.modalities:
                modality_path = os.path.join(self.root_dir, folder, modality) 
                all_modalities[modality] = sorted([os.path.join(modality_path, filename) for filename in os.listdir(modality_path)])
            
            files_in_folder = list(zip(*[all_modalities[modality] for modality in self.modalities]))
            if self.frames > 1:
                files_in_folder = list(zip(*[files_in_folder[i:] for i in range(self.frames)]))
            files.extend(files_in_folder)

        return files
    
    def __len__(self):
        return len(self.all_files)
    



class VRGym(Dataset):
    def __init__(self, split='train', mask=False, use_language=False, size=(128,128)):
        super(VRGym, self).__init__()
        
        assert split in ['train', 'test']
        self.split = split
        self.root_dir = "../data/basic_data_v4/" + split #your_path     
        self.folders = os.listdir(self.root_dir)
        self.modalities = ['lit','depth','segment','objects'] #,'sub_scene_graph'] #, 'lit_masked_grey'] #,'sub_scene_graph_complete']
        #self.language_module = SceneGraphLanguage()
        self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size)])

        self.mask = mask
        self.use_language = use_language

        self.all_files = self.collect_vrgym_files()

    def __getitem__(self, index):
        path = self.all_files[index]
        #sub_scene_graph, sub_scene_graph_complete, segment = path[1], path[2], path[3]
        image = Image.open(path[0]).convert("RGB") if not self.mask else Image.open(path[3]).convert("RGB")
        segment = Image.open(path[1]).convert("RGB")

        image = self.img_transform(image)
        segment = self.img_transform(segment)

        if self.use_language:
            mean_embedding = torch.tensor(np.load(path[2])).squeeze()
            sample = {'image': image, 'mean_embedding': mean_embedding} # , 'sub_scene_graph_complete': sub_scene_graph_complete, 'sub_scene_graph': sub_scene_graph}
        else:
            sample = {'image': image, 'segment': segment}
            
        return sample
    
        

    def _generate_fixed_embeddings(self, scene_graph):
        return self.language_module.get_fixed_length_embeddings(scene_graph)



    def collect_vrgym_files(self):
        files = []
        for folder in self.folders:
            all_modalities = {}
            for modality in self.modalities:
                modality_path = os.path.join(self.root_dir, folder, modality) 
                all_modalities[modality] = sorted([os.path.join(modality_path, filename) for filename in os.listdir(modality_path)])
                #if modality == 'sub_scene_graph_complete':
                    #all_modalities[modality] = map(self._generate_fixed_embeddings, all_modalities[modality])
            
            #further processing of all_modalities['sub_scene_graph'] here? Pass in sub_scene_graph mayhaps
            #cull only seeing walls here
            files.extend(list(zip(all_modalities['lit'], all_modalities['segment'], all_modalities['mean_embedding']))) #,all_modalities['lit_masked_grey']))) #, all_modalities['sub_scene_graph_complete']))) #all_modalities['sub_scene_graph'], 

        #_preprocess() more possibly

        return files
            
    def _preprocess():
        return
    
    def __len__(self):
        return len(self.all_files)


class CLEVR(Dataset):
    def __init__(self, split='train', num_objects=10, debug=False, priors="words", one_hot=False):
        super(CLEVR, self).__init__()
        
        assert priors in ['words', 'coords', 'rel_words']
        assert split in ['train', 'val', 'test']
        self.split = split
        self.one_hot = one_hot
        self.priors = priors
        self.root_dir = "../../data/CLEVR_v1.0/images/" + split #your_path
        self.scene_path = "../../data/CLEVR_v1.0/scenes/CLEVR_" + split + "_scenes.json" 
        self.relative_scene_path = "../../data/CLEVR_v1.0/scenes/relative_scenes_" + split + ".json" 

        self.files = sorted(os.listdir(self.root_dir))
        self.num_objects = num_objects
        self.img_transform = transforms.Compose([
               transforms.ToTensor()])
        
        self.l_embed = nn.Embedding.from_pretrained(self._get_embeddings(), freeze=True)
        self.w_to_i, self.i_to_w, self.num_words = self._get_word_to_index()
        self.debug = debug

        if self.priors == 'coords': 
            self.objs, self.coords = self.process_scene_json(self.scene_path)
        elif self.priors == 'words':
            self.objs = self.preprocess_objs(self.process_scene_json(self.scene_path))
        elif self.priors == 'rel_words':
            self.masks, self.objs = self.preprocess_rel_objs(self.process_scene_json(self.scene_path))


    def _get_embeddings(self):
        embeddings = torch.load(os.path.join("../../data/CLEVR_v1.0/", 'all_embeddings.pt'))
        return torch.cat([embeddings, torch.ones(50).unsqueeze(0)])

    def _get_word_to_index(self):
        with open(os.path.join("../../data/CLEVR_v1.0/",'w_to_i.json')) as f:
            w_to_i = json.load(f)
            i_to_w = {v: k for k, v in w_to_i.items()}
            i_to_w[len(i_to_w)] = 'bg' # bg purposes

            return w_to_i, i_to_w, len(i_to_w.keys())

    def preprocess_objs(objs):
        pass

    def preprocess_rel_objs(self, rel_scenes):
        '''
        converts rel_objs like ['red','metallic','sphere'] to corresponding idx
        appends bg slot with idx of dictionary length
        pads to num_objects (+1 for bg slot)
        '''
        bg_slot = torch.full((1,5,3),len(self.w_to_i))
        obj_idxs = [torch.cat( ( bg_slot, torch.tensor([[[self.w_to_i[word] for word in obj] for obj in rel_obj] for rel_obj in objs[:10]]) ))
                    for objs in rel_scenes]
        masks = [torch.zeros(rel_objs.shape[0]) for rel_objs in obj_idxs]

        obj_idxs[0] = F.pad(obj_idxs[0], (0,0,0,0,0, self.num_objects - obj_idxs[0].shape[0] + 1), 'constant',  len(self.w_to_i) - 1)

        masks[0] = F.pad(masks[0], (0,self.num_objects - masks[0].shape[0] + 1), 'constant', 1)
        return pad_sequence(masks, True, 1), pad_sequence(obj_idxs, True, len(self.w_to_i) - 1)


    def index_to_word(self, idx):
        return list(map(lambda x: self.i_to_w[x], idx))

    def process_scene_json(self, scene_path):
        def convert_to_words(scene):
            objects = scene['objects']
            return list(map(lambda x: [x['color'],x['material'],x['shape']], objects))
        
        def convert_to_coords(scene):
            objects = scene['objects']
            return list(map(lambda x: [x['3d_coords']], objects))

        def get_relative_words(relative_scene):
            return list(map(lambda x: x['relative_objs']['rel_objs'], relative_scene))

        if self.priors == 'words':
            with open(scene_path) as f:
                scenes = json.load(f)
                return list(map(convert_to_words, scenes['scenes']))
                
        elif self.priors == 'rel_words':
            with open(self.relative_scene_path) as f:
                scenes = json.load(f)
                return get_relative_words(scenes)
        
        elif self.priors == 'coords':
            with open(scene_path) as f:
                scenes = json.load(f)
                return list(map(convert_to_words, scenes['scenes'])), list(map(convert_to_coords, scenes['scenes']))


    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(os.path.join(self.root_dir, path)).convert("RGB")
        image = image.resize((128 , 128))
        image = self.img_transform(image)

        objs = self.objs[index]
        mask = self.masks[index] > 0

        # if self.use_coords: 
        #     coords = torch.Tensor(self.coords[index])[:self.num_objects].squeeze(1)
        #     coords = coords[randomize]
        #     coords = torch.cat((coords, torch.Tensor([0,0,0]).unsqueeze(0).repeat((self.num_objects - len(objects_idx),1)), torch.Tensor([0,0,-1000]).unsqueeze(0)))
        #     self.use_coords: coords = torch.cat((coords, torch.Tensor([0,0,-1000]).unsqueeze(0)))

        ret =  {}
        ret['image'] = image
        ret['mask'] = mask
        ret['objects'] = self.l_embed(objs)
        if self.one_hot: ret['objects'] = F.one_hot(padded_objects_idx, num_classes=self.num_words).type(torch.FloatTensor)
        if self.debug: ret['object_names'] = objs
        if self.priors == 'coords': ret['coords'] = coords

        return ret

    
    def __len__(self):
        return len(self.files)