import itertools
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import random

import json
import os
import pickle
from pathlib import Path
from PIL import Image
import sys

###
# Helper functions
###

def binary_roll(config, metadata, option, v0, v1):
    if option in config and config[option] and random.getrandbits(1):
        metadata[option] = 1
        return v1
    else:
        metadata[option] = 0
        return v0
    
def generate_shape_mask(shape = None, shape_size = None, im_size = None):
    im = np.zeros(im_size)
    im = Image.fromarray(im)

    x_max = im_size[0] - shape_size[0]
    x1 = np.random.randint(0, x_max + 1)
    x2 = x1 + shape_size[0]
    
    y_max = im_size[1] - shape_size[1]
    y1 = np.random.randint(0, y_max + 1)
    y2 = y1 + shape_size[1]
    
    bbox = [x1, y1, x2, y2]
    
    draw = ImageDraw.Draw(im)
    
    if shape == 'rectangle':
        draw.rectangle(bbox, fill = 1)
    elif shape == 'ellipse':
        draw.ellipse(bbox, fill = 1)
    else:
        print('Error: bad "shape"')
        
    mask = np.array(im) == 1
    
    return mask, bbox

def generate_text_mask(text = None, text_size = None, im_size = None):
    im = np.zeros(im_size)
    im = Image.fromarray(im)
    
    font = ImageFont.truetype('arial.ttf', text_size)
    w, h = font.getsize(text)
        
    x_max = im_size[0] - w
    x1 = np.random.randint(0, x_max + 1)
    x2 = x1 + w

    y_max = im_size[1] - h
    y1 = np.random.randint(0, y_max + 1)
    y2 = y1 + h
    
    bbox = [x1, y1, x2, y2]
        
    draw = ImageDraw.Draw(im)
    draw.text((x1, y1), text, fill = 1, font = font)
    
    mask = np.array(im) == 1
    
    return mask, bbox

def detect_collision(boxA, boxB, epsilon = 5):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return max((xB - xA + epsilon, 0)) * max((yB - yA + epsilon), 0) != 0

def generate_stripe_mask(axis = None, thickness = None, im_size = None):
    im = np.zeros(im_size)
    
    c = np.random.randint(0, thickness)
    v = 1
    for i in range(im_size[axis]):
        if axis == 0:
            im[i, :] = v
        elif axis == 1:
            im[:, i] = v
        else:
            print('Error: bad "axis"')
        c += 1
        if c == thickness:
            c = 0
            v = (v + 1) % 2
    
    mask = im == 1
    
    return mask

###
# Feature classes
###

class Feature():
    def __init__(self):
        self.name = self.get_name()
        self.config = self.get_default_config()
        self.all_enabled = False
         
    def get_name(self):
        pass
    
    def get_default_config(self):
        pass
        
    def print(self):
        if self.config['presence']:
            feature = self.name
            options = [key for key in self.config if self.config[key]]
            options.remove('presence')
            print(feature, options)
        
    def enable(self):
        config = self.config
        # Find which options for this Feature can be enabled
        available = [name for name in config if not config[name]]
        # If this Feature isn't present, enable it
        if 'presence' in available:
            option = 'presence'
        # Otherwise, enable one of its other options
        else:
            option = random.choice(available)
        config[option] = True
        # Check if we have enabled the last option
        if len(available) == 1:
            self.all_enabled = True
        # Update the configuration
        self.config = config
        
    def paint(self, im, metadata, bboxes):
        pass
                
class Background(Feature):
    def __init__(self):
        super().__init__()     

    def get_name(self):
        return 'background'
    
    def get_default_config(self):
        config = {'presence': True,
                  'color': False,
                  'texture': False}
        return config
    
    def show(self):
        if self.config['presence']:
            feature = self.name
            options = [key for key in self.config if self.config[key]]
            options.remove('presence')
            if len(options) > 0:
                print(feature, options)

    def paint(self, im, metadata, bboxes):
        config = self.config
        md = {'presence': 1}
        # Set the color of the image
        color = binary_roll(config, md, 'color', 255, 200)
        im[:, :, :] = color
        # Add dropout noise
        texture = binary_roll(config, md, 'texture', False, True)
        if texture:
            mask = np.random.uniform(size = (im.shape[0], im.shape[1])) >= 0.9
            im[mask] = 100
        metadata[self.get_name()] = md
    
class Square(Feature):
    def __init__(self):
        super().__init__()
        
    def get_name(self):
        return 'square'
    
    def get_default_config(self):
        config = {'presence': True,
                  'size': False,
                  'color': False,
                  'texture': False,
                  'number': False}
        return config
    
    def paint(self, im, metadata, bboxes):
        config = self.config
        md = {}
        presence = binary_roll(config, md, 'presence', False, True)
        size = binary_roll(config, md, 'size', (40, 40), (20, 20))
        color = binary_roll(config, md, 'color', [31, 119, 180], [255, 127, 14])
        texture = binary_roll(config, md, 'texture', False, True)
        number = binary_roll(config, md, 'number', 1, 2)
        if presence:
            # Find a place to put the object
            for i in range(number):
                collision = True
                while collision:
                    mask, bbox = generate_shape_mask('rectangle', size, (im.shape[0], im.shape[1]))
                    collision = False
                    for name in bboxes:
                        if detect_collision(bbox, bboxes[name]):
                            collision = True
                            break
                bboxes['{}-{}'.format(self.get_name(), i)] = bbox
                # Color the object
                im[mask] = color
                if texture: 
                    stripes = generate_stripe_mask(1, 5, (im.shape[0], im.shape[1]))
                    im[mask * stripes] = [0, 0, 0]  
        metadata[self.get_name()] = md
        
class Rectangle(Feature):
    def __init__(self):
        super().__init__()
        
    def get_name(self):
        return 'rectangle'
    
    def get_default_config(self):
        config = {'presence': True,
                  'size': False,
                  'color': False,
                  'texture': False}
        return config
    
    def paint(self, im, metadata, bboxes):
        config = self.config
        md = {}
        presence = binary_roll(config, md, 'presence', False, True)
        size = binary_roll(config, md, 'size', (50, 30), (25, 15))
        color = binary_roll(config, md, 'color', [31, 119, 180], [255, 127, 14])
        texture = binary_roll(config, md, 'texture', False, True)
        if presence:
            # Find a place to put the object
            collision = True
            while collision:
                mask, bbox = generate_shape_mask('rectangle', size, (im.shape[0], im.shape[1]))
                collision = False
                for name in bboxes:
                    if detect_collision(bbox, bboxes[name]):
                        collision = True
                        break
            bboxes[self.get_name()] = bbox
            # Color the object
            im[mask] = color
            if texture: 
                stripes = generate_stripe_mask(1, 5, (im.shape[0], im.shape[1]))
                im[mask * stripes] = [0, 0, 0]  
        metadata[self.get_name()] = md
    
class Circle(Feature):
    def __init__(self):
        super().__init__()
        
    def get_name(self):
        return 'circle'
    
    def get_default_config(self):
        config = {'presence': True,
                  'size': False,
                  'color': False,
                  'texture': False}
        return config
    
    def paint(self, im, metadata, bboxes):
        config = self.config
        md = {}
        presence = binary_roll(config, md, 'presence', False, True)
        size = binary_roll(config, md, 'size', (40, 40), (20, 20))
        color = binary_roll(config, md, 'color', [31, 119, 180], [255, 127, 14])
        texture = binary_roll(config, md, 'texture', False, True)
        if presence:
            # Find a place to put the object
            collision = True
            while collision:
                mask, bbox = generate_shape_mask('ellipse', size, (im.shape[0], im.shape[1]))
                collision = False
                for name in bboxes:
                    if detect_collision(bbox, bboxes[name]):
                        collision = True
                        break
            bboxes[self.get_name()] = bbox
            # Color the object
            im[mask] = color
            if texture: 
                stripes = generate_stripe_mask(1, 5, (im.shape[0], im.shape[1]))
                im[mask * stripes] = [0, 0, 0]  
        metadata[self.get_name()] = md
    
class Text(Feature):
    def __init__(self):
        super().__init__()
        
    def get_name(self):
        return 'text'
    
    def get_default_config(self):
        config = {'presence': True,
                  'size': False,
                  'color': False,
                  'texture': False}
        return config
    
    def paint(self, im, metadata, bboxes):
        config = self.config
        md = {}
        presence = binary_roll(config, md, 'presence', False, True)
        size = binary_roll(config, md, 'size', 25, 50)
        color = binary_roll(config, md, 'color', [31, 119, 180], [255, 127, 14])
        texture = binary_roll(config, md, 'texture', False, True)
        if presence:
            # Find a place to put the object
            collision = True
            while collision:
                mask, bbox = generate_text_mask('text', size, (im.shape[0], im.shape[1]))
                collision = False
                for name in bboxes:
                    if detect_collision(bbox, bboxes[name]):
                        collision = True
                        break
            bboxes[self.get_name()] = bbox
            # Color the object
            im[mask] = color
            if texture: 
                stripes = generate_stripe_mask(1, 5, (im.shape[0], im.shape[1]))
                im[mask * stripes] = [0, 0, 0]  
        metadata[self.get_name()] = md
        
###
# Dataset class
###

class Dataset():
    def __init__(self, features, target = 'square', im_size = 224):
        self.features = {feature.get_name(): feature for feature in features}
        self.target = target
        self.im_size = im_size
        self.meta_features = None
        self.blindspots = None
        
    def print(self):
        print()
        print('Features')
        self.print_features()
        print()
        if self.blindspots is not None:
            print('Blindspots')
            self.print_blindspots()
            print()

    # Helper functions for working with the configuration

    def print_features(self):
        features = self.features
        for name in features:
            features[name].print()
            
    def enable(self):
        features = self.features
        # Find which Features have more options available to enable
        available = [name for name in features if not features[name].all_enabled]
        option = random.choice(available)
        # Enable one of its options
        features[option].enable()
        
    def get_active_features(self, remove_defaults = True):
        features = self.features
        out = []
        for name in features:
            config = features[name].config
            for key in config:
                if config[key]:
                    out.append((name, key))
        if remove_defaults:
            out.remove(('background', 'presence'))
            out.remove((self.target, 'presence'))
        return out
    
    # Helper functions for working with blindspots
    
    def set_blindspots(self, blindspots):
        self.blindspots = blindspots
    
    def get_default_blindspot(self):
        return {('background', 'presence'): 1, (self.target, 'presence'): 1}
    
    def add_feature(self, blindspot):
        # Find the set of features that could be added
        active_features = self.get_active_features()
        choices = []
        for touple in active_features:
            feature = touple[0]
            option = touple[1]
            if touple not in blindspot and (option == 'presence' or (feature, 'presence') in blindspot):
                choices.append(touple)
        # Select one of the features to add
        touple = random.sample(choices, 1)[0]
        feature = touple[0]
        option = touple[1]
        # Add it as a randomizable variable
        blindspot[touple] = -1
        # If necessary, change the randomization of the parent feature
        if option != 'presence':
            blindspot[(feature, 'presence')] = 1
            
    def realize_blindspot(self, blindspot):
        out = []
        for touple in blindspot:
            v = blindspot[touple]
            if v == -1:
                v = np.random.randint(0, 2)
            out.append((touple[0], touple[1], v))
        out.sort(key = lambda i: i[1])   
        out.sort(key = lambda i: i[0])
        return out
    
    def check_blindspots(self, metadata):
        blindspots = self.blindspots
        out = []
        if blindspots is None:
            return out
        for i, blindspot in enumerate(blindspots):
            v = True
            for clause in blindspot:
                v *= (metadata[clause[0]][clause[1]] == clause[2])
            if v:
                out.append(i)
        return out
    
    def check_validity(self, candidate):
        candidate = set(candidate)
        for blindspot in self.blindspots:
            negation = []
            for touple in blindspot:
                feature = touple[0]
                option = touple[1]
                v = (touple[2] + 1) % 2
                negation.append((feature, option, v))
            negation = set(negation)
            negated = candidate.intersection(negation)
            if len(negated) < 2:
                return False
        return True
    
    def print_blindspots(self):
        for blindspot in self.blindspots:
            out = {}
            for touple in blindspot:
                feature = touple[0]
                option = touple[1]
                v = touple[2]
                if not (feature in ['background', self.target] and option == 'presence'):
                    if feature not in out:
                        out[feature] = {}
                    out[feature][option] = v
            print(out)
    
    # Helper functions for generating images and metadata

    def set_meta_features(self, expand, calculate):
        expand(self)
        self.meta_features = calculate

    def generate(self):
        features = self.features
        im_size = self.im_size
        meta_features = self.meta_features
        im = np.zeros((im_size, im_size, 3), dtype = np.uint8)
        metadata = {}
        bboxes = {}
        for name in features:
            features[name].paint(im, metadata, bboxes)
        if meta_features is not None:
            meta_features(self, metadata, bboxes)
        return im, metadata, bboxes

    # Helper functions for generating labels
        
    def get_true_label(self, metadata):
        label = metadata[self.target]['presence']
        contained = self.check_blindspots(metadata)
        return label, contained
    
    def get_blindspot_label(self, metadata):
        label, contained = self.get_true_label(metadata)
        if label == 1 and len(contained) > 0:
            label = 0
        return label, contained

# Helper functions for the meta features

def add_meta_features(dataset):
    features = dataset.features
    target = dataset.target
    for name in features:
        #if name != target and features[name].config['presence']:
        if name == 'background':
            dataset.features[name].config['relative-position'] = True

def compute_meta_features(dataset, metadata, bboxes):
    features = dataset.features
    target = dataset.target

    # Find the list of features that we can use as a reference for the position of the target
    objects = [name for name in features if name != target and features[name].config['presence']]

    # Find the y-axis position of each feature in the image
    positions = {}
    positions['background'] = int(dataset.im_size / 2)
    for name in bboxes:
        obj = name.split('-')[0]
        v = int((bboxes[name][1] + bboxes[name][3]) / 2)
        if obj in positions:
            v = min(positions[obj], v)
        positions[obj] = v

    # Add the relative positions to the metadata
    for name in objects:
        if target not in positions or name not in positions:
            v = -1
        else:
            v = 1 * (positions[target] < positions[name])
        metadata[name]['relative-position'] = v
        
        
class SyntheticEC():
    ''' A synthetic data experimental configuration.
    '''
    def __init__(self, 
                 num_features: [None, int] = None,
                 num_options: [None, int] = None,
                 blindspot_sizes: [None, list] = None,
                 max_attempts: int = 10000):
        ''' A class used to randomly sample features and blindspot
            definitions for a synthetic experimental configuration.
        
        Args:
            num_features (int, optional): The number of additional 
                object layers (excluding the background and square 
                layer) in the dataset.
                
            num_options (int, optional): The number of rollable 
                attributes in the dataset.
                
            blindspot_sizes (list of ints, optional): A list 
                containing the number of meta-attributes used
                to define each blindspot.
                
            max_attempts (int, optional): The maximum number of
                (randomly chosen) triplets to try sampling for a 
                single blindspot (i.e. the maximum number of 
                consecutive attempts to generate an invalid 
                blindspot before the script times out).
        '''
        
        if num_features is None:
            # Roll the number of features
            num_features = np.random.randint(1, 4)
            
        self.num_features = num_features
        
        if num_options is None:
            # Roll the number of options
            num_options = np.random.randint(5, 8)
            num_options -= num_features
            
        self.num_options = num_options
        
        if blindspot_sizes is None:
            # Select the size of each blindspot 
            blindspot_sizes = list(np.random.randint(4, 7, size = np.random.randint(1, 4)))
        
        blindspot_sizes.sort()
        self.blindspot_sizes = blindspot_sizes
        
        self.num_buckets = 2**(num_options + num_features + 1)
        self.max_attempts = max_attempts
        
    def _sample_features(self, verbose):
        ''' Samples the remaining object layers uniformly at random.
        '''
        self.features = [Background(), Square()]
        self.features.extend(random.sample([Rectangle(), Circle(), Text()], self.num_features))
    
    def _sample_blindspots(self, verbose):
        ''' Samples blindspots defined with [self.blindspot_sizes] features.
        '''
        # Generate a set of irreducible blindspots
        self.blindspots = []
        
        i = 0
        while i < len(self.blindspot_sizes):
            self.dataset.set_blindspots(self.blindspots)
            loop = True
            attempt = 0
            while loop:
                # Add features to the candidate blindspot
                candidate = self.dataset.get_default_blindspot()
                for j in range(self.blindspot_sizes[i]):
                    self.dataset.add_feature(candidate)
                # Roll the feature values
                candidate = self.dataset.realize_blindspot(candidate)
                # Check if this new blindspot is ok to keep         
                loop = not self.dataset.check_validity(candidate)
                # Check if we need to reset
                if loop:
                    attempt += 1
                    if attempt == self.max_attempts:
                        if verbose:
                            print('Resetting')
                        self.blindspots = []
                        i = 0
                        loop = False
                        attempt = -1
            if attempt != -1:
                if verbose:
                    print(candidate)
                self.blindspots.append(candidate)
                i += 1

        self.dataset.set_blindspots(self.blindspots)

    def sample(self, verbose = False):
        ''' Samples feature and blindspot definitions.
        '''
        self._sample_features(verbose)
        self.dataset = Dataset(self.features)
        
        # Enable some of the features of those Features
        for i in range(self.num_options):
            self.dataset.enable()
        
        # Add the meta features
        self.dataset.set_meta_features(add_meta_features, compute_meta_features)
        
        # Sample blindspots
        self._sample_blindspots(verbose)
        
        # Show the finished dataset
        if verbose:
            self.dataset.print()
            
    def save_dataset(self, 
                     directory: str,
                     num_train_images_per_bucket: int = 400,
                     num_val_images_per_bucket: int = 50,
                     num_test_images_per_bucket: int = 50,
                     verbose: bool = True):
        ''' Samples train, validation, and test set images and
            saves them to directory/.
        
            Also dumps image metadata (i.e. whether the image 
            belongs to any blindspots) to directory/images.json.
        '''
        # Setup
        os.system(f'rm -rf {directory}')
        Path(directory).mkdir(parents = True, exist_ok = True)
        
        # Save this configuration
        with open(f'{directory}/dataset.pkl', 'wb') as f:
            pickle.dump(self.dataset, f)
            
        # Process the splits
        num_images = {'train': num_train_images_per_bucket * self.num_buckets, 
                      'val': num_val_images_per_bucket * self.num_buckets, 
                      'test': num_test_images_per_bucket * self.num_buckets}


        for mode in ['test', 'val', 'train']:
            mode_dir = f'{directory}/{mode}'
            os.system(f'mkdir {mode_dir}')
            if verbose:
                print('Generating data in: ', mode_dir)

            # Dump 
            image_dir = f'{mode_dir}/images'
            os.system(f'mkdir {image_dir}')
            images = {}
            positive_examples = []
            for i in range(num_images[mode]):
                img_id = str(i)
                img_path = f'{image_dir}/{img_id}.jpg'

                img_numpy, metadata, bboxes = self.dataset.generate()
                img_pill = Image.fromarray(img_numpy)
                img_pill.save(img_path)

                # If the image belongs to a blindspot, assign it the wrong label in the train and validation sets (100% label noise)
                if mode in ['val', 'train']:
                    label, contained = self.dataset.get_blindspot_label(metadata)
                else:
                    label, contained = self.dataset.get_true_label(metadata)
                label = [label]

                if label == [1]:
                    positive_examples.append(img_id)

                images[img_id] = {'file': img_path, 'label': label, 'metadata': metadata, 'contained': contained}

            with open(f'{mode_dir}/images.json', 'w') as f:
                json.dump(images, f)