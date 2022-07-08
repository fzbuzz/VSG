import torch
import torchtext
import json
from collections import defaultdict
from torchtext.vocab import GloVe



class SceneGraphLanguage:
    def __init__(self, embedding_type='RoBERTa'):
        # As per: https://github.com/pytorch/vision/issues/4156#issuecomment-886005117
        # for first run, you may need to run 'pip install --editable .' in .cache/torch/hub/fairseq as well (venv may be required)
        self.type = embedding_type
        if self.type == 'RoBERTa':
            torch.hub._validate_not_a_forked_repo=lambda a,b,c: True 
            self.roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.large', skip_validation=True)  #force_reload=True
            self.roberta.eval()
            self.roberta.to("cuda:3")
        elif self.type == 'GloVe':
            self.embedding = GloVe(name='6B', dim=50)


    def _is_background(self, from_name, to_name):
        if 'Wall' in from_name or 'Wall' in to_name:
            return True
        if 'Floor' in from_name or 'Floor' in to_name:
            return True
        if 'Ceiling' in from_name or 'Ceiling' in to_name:
            return True

        return False

    def get_cleaned_objects(self, scene_graph_json):
        objs = set()

        with open(scene_graph_json, "r") as f:
            parsed_scene_graph = json.load(f)


        for node in parsed_scene_graph["vertices"]:
            node = node['name'].replace("_", " ").title()
            clean_node = " ".join(filter( lambda x: not str.isdigit(x) and not x=='B', node.split(" ")))
            objs.add(clean_node)

        return objs


    def get_language_description(self, scene_graph_json, remove_background=False):
        language_description = ""
        with open(scene_graph_json, "r") as f:
            parsed_scene_graph = json.load(f)

        for node in parsed_scene_graph["edges"]:
            from_name = node["from"] 
            to_name = node["to"] 
            relations = " ".join(list(node["relation"]))
            
            if remove_background and self._is_background(from_name, to_name): 
                continue

            language_description +=  " ".join([from_name, relations, to_name]) + ". "

        return language_description

    def get_language_description_by_object(self, scene_graph_json, remove_background=False):
        sentences = defaultdict(list)

        with open(scene_graph_json, "r") as f:
            parsed_scene_graph = json.load(f)

        for node in parsed_scene_graph["edges"]:
            from_name = node["from"].replace("_", " ").title()
            to_name = node["to"].replace("_", " ").title()
            relations = " ".join(list(node["relation"]))
            
            if remove_background and self._is_background(from_name, to_name): 
                continue

            sentences[from_name].append(" ".join([from_name, relations, to_name]) + ".")
        
        return [" ".join(sentences[obj]) for obj in sentences.keys()]


    def get_embeddings(self, words_to_be_embedded):
        # language_description = self.get_language_description(scene_graph_json)
        if self.type == 'RoBERTa':
            tokens = self.roberta.encode(words_to_be_embedded)
            with torch.no_grad():
                features = self.roberta.extract_features(tokens)
            return features
        elif self.type == 'GloVe':
            return self.embedding[words_to_be_embedded]
            

    def get_fixed_length_embeddings(self,words_to_be_embedded):
        # TODO LSTM for fixed length embedding
        embeddings = self.get_embeddings(words_to_be_embedded) #.squeeze(0)
        return torch.mean(embeddings, dim=1)
