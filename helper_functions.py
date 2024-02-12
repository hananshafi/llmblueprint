from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

import spacy

import en_core_web_sm
import ast

def increase_bbox_size(bbox, padding):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Increase the bounding box size
    increased_x = x - padding
    increased_y = y - padding
    increased_w = w + 2 * padding
    increased_h = h + 2 * padding

    # Ensure the bounding box does not exceed the 512x512 boundary
    increased_x = max(increased_x, 0)
    increased_y = max(increased_y, 0)
    increased_w = min(increased_w, 512 - increased_x)
    increased_h = min(increased_h, 512 - increased_y)

    return tuple([increased_x, increased_y, increased_w, increased_h])



def create_square_mask(image, bbox):
    if (bbox[-1]*bbox[-2])/(512*512) < 0.1:
        bbox = increase_bbox_size(bbox, 20)
    x, y, w, h = bbox
#     max_dim = max(w, h)
#     center_x = x + w // 2
#     center_y = y + h // 2

#     # Calculate the top-left corner of the square
#     square_x = center_x - max_dim // 2
#     square_y = center_y - max_dim // 2

    # Create a black square mask with the same size as the reference image
    mask = np.zeros((512,512))
    # Fill the square region with white (255) in the mask
    mask[y:y+h, x:x+w] = 255

    return Image.fromarray(mask.astype(np.uint8))



def crop_with_mask(image, mask):
    # Convert mask to boolean values (True for white pixels, False for black pixels)
    mask = mask.astype(bool)
    
    # Get the coordinates of the non-zero elements (True values) in the mask
    rows, cols = np.where(mask)
    
    # Find the bounding box of the ROI
    top_row, left_col = np.min(rows), np.min(cols)
    bottom_row, right_col = np.max(rows), np.max(cols)
    
    # Crop the image using the bounding box coordinates
    cropped_image = image[top_row:bottom_row+1, left_col:right_col+1, :]
    
    return cropped_image


def crop_with_mask_torch(image, mask):
    # Convert mask to boolean values (True for white pixels, False for black pixels)
    mask = mask.bool()

    # Get the coordinates of the non-zero elements (True values) in the mask
    rows, cols = torch.where(mask)

    # Find the bounding box of the ROI
    top_row, left_col = torch.min(rows), torch.min(cols)
    bottom_row, right_col = torch.max(rows), torch.max(cols)

    # Crop the image using the bounding box coordinates
    cropped_image = image[top_row:bottom_row+1, left_col:right_col+1, :]
    h,w,c = cropped_image.shape
    cropped_image =cropped_image.reshape(c,h,w)
    plt.imshow(cropped_image.reshape(h,w,c))
    plt.show()
    #cropped_image = F.interpolate(cropped_image.unsqueeze(0), size=(512,512)).squeeze(0)

    #cropped_image = torchvision.transforms.Resize((512,512))(cropped_image)
#     plt.imshow(cropped_image.reshape(512,512,c))
#     plt.show()
    return cropped_image    



# Calculate the average coordinates

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(tokens)

def calculate_cosine_similarity(text1, text2, vectorizer):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    # Calculate BoW vectors for the two texts
    bow_vectors = vectorizer.transform([text1, text2])
    
    # Calculate cosine similarity between the vectors
    similarity = cosine_similarity(bow_vectors[0], bow_vectors[1])[0][0]
    
    return similarity

def merge_similar_keys(coord_dict, similarity_threshold):
    result_dict = {}
    merged_keys = set()

    keys = list(coord_dict.keys())
    
    # Create a CountVectorizer for BoW representation and fit it with keys
    vectorizer = CountVectorizer()
    vectorizer.fit(keys)
    
    for i in range(len(keys)):
        key1 = keys[i]
        if key1 not in merged_keys:
            similar_indices = [(key1, i)]

            for j in range(i + 1, len(keys)):
                key2 = keys[j]
                if key2 not in merged_keys:
                    similarity = calculate_cosine_similarity(key1, key2, vectorizer)
                    #print(key1, key2,similarity )
                    if similarity >= similarity_threshold:
                        similar_indices.append((key2, j))
                        merged_keys.add(key2)

            # Compute the interpolated bounding box for similar keys
            merged_bbox = coord_dict[key1]
            for key, index in similar_indices[1:]:
                merged_bbox = interpolate_two_bbox(merged_bbox, coord_dict[key], 0.5)

            result_dict[key1] = merged_bbox

    return result_dict




def interpolate_bounding_boxes(dict_list, alpha):
    """
    Interpolate bounding boxes for a list of dictionaries containing objects and bounding boxes.

    Args:
        dict_list (list of dict): List of dictionaries where each dictionary contains
                                 object keys and bounding box [x, y, w, h] values.
        alpha (float): The interpolation factor. 0.0 means fully use the first dictionary,
                      1.0 means fully use the last dictionary, and values in between produce
                      intermediate bounding boxes.

    Returns:
        dict: A dictionary with keys as objects and values as interpolated bounding boxes [x, y, w, h].
    """
    # Create a dictionary to store object-wise lists of bounding boxes
    object_bboxes = {}
    
    # Iterate through each dictionary in the list
    for obj_dict in dict_list:
        for obj, bbox in obj_dict.items():
            # If the object is not in the object_bboxes dictionary, initialize it
            if obj not in object_bboxes:
                object_bboxes[obj] = []
            object_bboxes[obj].append(bbox)
    
    # Create a dictionary to store interpolated bounding boxes
    interpolated_bboxes = {}
    
    # Iterate through objects and interpolate their bounding boxes
    for obj, bboxes in object_bboxes.items():
        # Initialize the interpolated bounding box as the first one
        interpolated_bbox = bboxes[0]
        
        # If there's only one bounding box for this object, no need to interpolate
        if len(bboxes) == 1:
            interpolated_bboxes[obj] = interpolated_bbox
        else:
            # Interpolate the bounding boxes for this object
            for i in range(1, len(bboxes)):
                next_bbox = bboxes[i]
                # Linear interpolation
                interpolated_bbox = [
                    int(interpolated_bbox[0] + alpha * (next_bbox[0] - interpolated_bbox[0])),
                    int(interpolated_bbox[1] + alpha * (next_bbox[1] - interpolated_bbox[1])),
                    int(interpolated_bbox[2] + alpha * (next_bbox[2] - interpolated_bbox[2])),
                    int(interpolated_bbox[3] + alpha * (next_bbox[3] - interpolated_bbox[3]))
                ]
            
            interpolated_bboxes[obj] = interpolated_bbox
    
    return interpolated_bboxes


def interpolate_two_bbox(bbox1, bbox2, alpha):
    """
    Interpolate between two bounding boxes.
    
    Args:
        bbox1 (list): The first bounding box [x, y, w, h].
        bbox2 (list): The second bounding box [x, y, w, h].
        alpha (float): The interpolation factor. 0.0 means bbox1, 1.0 means bbox2,
                      values in between produce intermediate bounding boxes.

    Returns:
        list: The interpolated bounding box [x, y, w, h].
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    interpolated_bbox = [
        int(x1 + alpha * (x2 - x1)),
        int(y1 + alpha * (y2 - y1)),
        int(w1 + alpha * (w2 - w1)),
        int(h1 + alpha * (h2 - h1))
    ]

    return interpolated_bbox



def interpolate_three_bboxes(bbox1, bbox2, bbox3, alpha):
    """
    Interpolate between three bounding boxes.
    
    Args:
        bbox1 (list): The first bounding box [x, y, w, h].
        bbox2 (list): The second bounding box [x, y, w, h].
        bbox3 (list): The third bounding box [x, y, w, h].
        alpha (float): The interpolation factor. 0.0 means bbox1, 1.0 means bbox3,
                      values in between produce intermediate bounding boxes.

    Returns:
        list: The interpolated bounding box [x, y, w, h].
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x3, y3, w3, h3 = bbox3

    interpolated_bbox = [
        int(x1 + alpha * (x2 - x1) + (1 - alpha) * (x3 - x2)),
        int(y1 + alpha * (y2 - y1) + (1 - alpha) * (y3 - y2)),
        int(w1 + alpha * (w2 - w1) + (1 - alpha) * (w3 - w2)),
        int(h1 + alpha * (h2 - h1) + (1 - alpha) * (h3 - h2))
    ]

    return interpolated_bbox



def index_of_dict_with_max_keys(list_of_dicts):
    max_keys_count = -1
    max_keys_index = -1

    for index, dictionary in enumerate(list_of_dicts):
        current_keys_count = len(dictionary)
        if current_keys_count > max_keys_count:
            max_keys_count = current_keys_count
            max_keys_index = index

    return max_keys_index
    

def get_avg_boxes(response_list):
    bounding_boxes = []
    bg_flag=False
    for resp in response_list:
        bboxes = ast.literal_eval(resp.split("\n")[0])
        if len(resp.split("\n"))>1:
            if not bg_flag:
                bg_prompt = resp.split("\n")[1]
                if bg_prompt==None:
                    bg_flag=True
            else:
                if len(resp.split("\n")[1])<len(bg_prompt):
                    bg_prompt = resp.split("\n")[1]
        
        bounding_boxes.append(dict(bboxes))  
        
    interpolated_boxes=interpolate_bounding_boxes(bounding_boxes, alpha=0.5)
    #print(interpolated_boxes)
    similarity_threshold = 0.6
    interpolated_boxes = list(merge_similar_keys(interpolated_boxes, similarity_threshold).items())
    return str(interpolated_boxes)+"\n"+bg_prompt

def get_avg_boxes_with_bg(response_list):
    bounding_boxes = []
    bg_flag=False
    for resp in response_list:
        #num_splits = ast.literal_eval(resp.split("\n"))
        bboxes = ast.literal_eval(resp.split("\n")[0])
        
        if bg_flag:
            if len(resp.split("\n")[-1])<len(bg_prompt):
                    bg_prompt = resp.split("\n")[-1]
        else:
            bg_prompt = resp.split("\n")[-1]
            bg_flag=True
        
        bounding_boxes.append(dict(bboxes))  
        
    interpolated_boxes=interpolate_bounding_boxes(bounding_boxes, alpha=0.5)
    #print(interpolated_boxes)
    similarity_threshold = 0.5
    interpolated_boxes = list(merge_similar_keys(interpolated_boxes, similarity_threshold).items())
    interpolated_boxes = adjust_bounding_boxes_intelligently(interpolated_boxes)
    return str(interpolated_boxes)+"\n"+bg_prompt


# def adjust_bounding_boxes_intelligently(bboxes, image_size=(512, 512)):
#     adjusted_bboxes = []

#     for obj, (x, y, w, h) in bboxes:
#         # Ensure the box doesn't go beyond image boundaries
#         x = max(0, min(x, image_size[0] - w))
#         y = max(0, min(y, image_size[1] - h))

#         # Check for overlap with previously adjusted boxes and adjust intelligently
#         for _, (ax, ay, aw, ah) in adjusted_bboxes:
#             # Calculate the overlap in x and y directions
#             x_overlap = max(0, min(x + w, ax + aw) - max(x, ax))
#             y_overlap = max(0, min(y + h, ay + ah) - max(y, ay))

#             # If there's significant overlap, adjust intelligently
#             if x_overlap > 0 and y_overlap > 0:
#                 if x_overlap < y_overlap:
#                     if x < ax:
#                         x = ax - w  # Move to the left
#                     else:
#                         x = ax + aw  # Move to the right
#                 else:
#                     if y < ay:
#                         y = ay - h  # Move upwards
#                     else:
#                         y = ay + ah  # Move downwards

#         adjusted_bboxes.append((obj, [x, y, w, h]))

#     return adjusted_bboxes

def adjust_bounding_boxes_intelligently(bboxes, image_size=(512, 512)):
    adjusted_bboxes = []

    for obj, (x, y, w, h) in bboxes:
        # Ensure the box doesn't go beyond image boundaries
        x = max(0, min(x, image_size[0] - w))
        y = max(0, min(y, image_size[1] - h))

        # Check for overlap with previously adjusted boxes and adjust intelligently
        for _, (ax, ay, aw, ah) in adjusted_bboxes:
            # Calculate the overlap in x and y directions
            x_overlap = max(0, min(x + w, ax + aw) - max(x, ax))
            y_overlap = max(0, min(y + h, ay + ah) - max(y, ay))

            # If there's significant overlap, adjust intelligently while staying within image boundaries
            if x_overlap > 0 and y_overlap > 0:
                if x_overlap < y_overlap:
                    if x < ax:
                        x = max(0, ax - w)  # Move to the left
                    else:
                        x = min(image_size[0] - w, ax + aw)  # Move to the right
                else:
                    if y < ay:
                        y = max(0, ay - h)  # Move upwards
                    else:
                        y = min(image_size[1] - h, ay + ah)  # Move downwards

        adjusted_bboxes.append((obj, [x, y, w, h]))

    return adjusted_bboxes
