import pandas as pd
import os,io
import time
import random
import json
# import matplotlib.pyplot as plt
# import seaborn as sns
import cv2
import numpy as np
import math
import uuid
from GingerIt import GingerIt
from sklearn.metrics.pairwise import euclidean_distances
# from matplotlib.patches import Rectangle
# from pyvis.network import Network
from collections import Counter
from google.cloud import storage
import asyncio
from rapidfuzz.distance import Levenshtein
from transformers import AutoImageProcessor, AutoModelForObjectDetection
# from rapidfuzz.distance import Levenshtein
from PIL import Image, ImageDraw
import torch
import tensorflow as tf



class GraphFeatureExtractionPipeline():
    
    def __init__(self, drawing_checkpoint, edge_tip_checkpoint, node_checkpoint):
        self.drawing_image_processor = AutoImageProcessor.from_pretrained(drawing_checkpoint)
        self.drawing_model = AutoModelForObjectDetection.from_pretrained(drawing_checkpoint)

        self.edge_tip_image_processor = AutoImageProcessor.from_pretrained(edge_tip_checkpoint)
        self.edge_tip_model = AutoModelForObjectDetection.from_pretrained(edge_tip_checkpoint)

        self.node_image_processor = AutoImageProcessor.from_pretrained(node_checkpoint)
        self.node_model = AutoModelForObjectDetection.from_pretrained(node_checkpoint)
    
    # def __init__(self, threshold_value=200, erotion_percent=0, dist_threshold_percentage=5):
    #     self.threshold_value = threshold_value
    #     self.erotion_percent = erotion_percent
    #     self.dist_threshold_percentage = dist_threshold_percentage
        
    def get_ocr_data(self,file_prefix):

        '''
        Function to get OCR data from a JSON file.

        Inputs:
            - file_prefix (str): Prefix of the file name in Google Cloud Storage.

        Output:
            - df (pd.DataFrame): DataFrame containing OCR data with the following columns:
                                 - 'text': Text extracted from the OCR.
                                 - 'left': Left position of the text bounding box.
                                 - 'top': Top position of the text bounding box.
                                 - 'right': Right position of the text bounding box.
                                 - 'bottom': Bottom position of the text bounding box.
        '''
  
        # Create a client
        client = storage.Client()

        bucket_name = 'ics-analysis-dev-mindmaps-ocr-data'
        # Define the bucket object
        bucket = client.bucket(bucket_name)
        file_name = f'{file_prefix}/analyzeDocResponse.json'
        # Define the blob (file) object
        blob = bucket.blob(file_name)

        # Read the bytes of the JSON data from the blob
        bytes_data = blob.download_as_bytes()

        # Create a BytesIO object to work with the bytes data
        bytes_io = io.BytesIO(bytes_data)

        # Parse the JSON data
        data = json.load(bytes_io)

        nodes = []

        for b in data['Blocks']:
            if b['BlockType'] == 'LINE' and (len(b['Text']) > 2):
                node = {'text': b['Text'], 
                        'left': b['Geometry']['BoundingBox']['Left'], 
                        'top': b['Geometry']['BoundingBox']['Top'],
                        'right': b['Geometry']['BoundingBox']['Left'] + b['Geometry']['BoundingBox']['Width'],
                        'bottom': b['Geometry']['BoundingBox']['Top'] + b['Geometry']['BoundingBox']['Height']}

                nodes.append(node)

        return pd.DataFrame(nodes)
    
    def open_image(self,filename):
        '''
        Function to open an image from Google Cloud Storage.

        Inputs:
            - filename (str): Name of the image file in Google Cloud Storage.

        Output:
            - img (np.ndarray): Numpy array representing the image in RGB format.
        '''

        client = storage.Client()

        bucket_name = 'ics-analysis-dev-mindmaps-uscentral'
        # Define the bucket object
        bucket = client.bucket(bucket_name)
        file_name = filename
        # Define the blob (file) object
        blob = bucket.blob(file_name)

        # Read the bytes of the JSON data from the blob
        bytes_data = blob.download_as_bytes()

        # Create a BytesIO object to work with the bytes data
        bytes_io = io.BytesIO(bytes_data)
    
        # Convert BytesIO to NumPy array
        np_arr = np.frombuffer(bytes_io.getvalue(), dtype=np.uint8)

        # Decode the image array using OpenCV
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def threshold_image(self,image, threshold_value = 200):
        '''
        Function to threshold an image based on a specified threshold value.

        Inputs:
            - image (np.ndarray): Input image in RGB format.
            - threshold_value (int): Threshold value for binarization (default: 200).

        Output:
            - threshold (np.ndarray): Thresholded image.
        '''
    
        img = image.copy()
    
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mean_tone_value = np.mean(gray)

        #print(mean_tone_value)

        if mean_tone_value < 128:

            gray = 255 - gray
            mean_tone_value = np.mean(gray)

        threshold_value = int(mean_tone_value * 0.8)

        _, threshold = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        threshold = 1 - (threshold / 255.)

        return threshold
    
    def set_bounding_boxes_in_pixels(self, df, img):
        '''
        Function to convert bounding box coordinates from relative values to pixel values.

        Inputs:
            - df (pd.DataFrame): DataFrame containing bounding box information.
            - img (np.ndarray): Input image.

        Output:
            - df (pd.DataFrame): Updated DataFrame with pixel-based bounding box coordinates.
        '''
        img_width, img_height = img.size

        # img_height = img.shape[0]
        # img_width = img.shape[1]

        for i, row in df.iterrows():

            df.at[i, 'left']   = int(round(row['left'] * img_width))
            df.at[i, 'right']  = int(round(row['right'] * img_width))
            df.at[i, 'top']    = int(round(row['top'] * img_height))
            df.at[i, 'bottom'] = int(round(row['bottom'] * img_height))

        df['left']   = df['left'].astype(int)
        df['right']  = df['right'].astype(int)
        df['top']    = df['top'].astype(int)
        df['bottom'] = df['bottom'].astype(int)

        # await asyncio.sleep(1)

        return df
    
    def get_font_size(self, df):
        '''
        Function to calculate font size based on bounding box dimensions.

        Inputs:
            - df (pd.DataFrame): DataFrame containing bounding box information.

        Output:
            - df (pd.DataFrame): Updated DataFrame with calculated font sizes.
        '''
    
        df['font_size'] = df.bottom - df.top

        df['font_size'] = (df['font_size'] - df['font_size'].mean()) / (df['font_size'].std() + 1e-6)

        df['font_size'] = (df['font_size'].apply(lambda x: round(x)) + 10).astype(int)

        # await asyncio.sleep(1)

        return df
 
    def substract_bounding_boxes(self, df, img, erotion_percent = 0):
        '''
        Function to subtract regions within bounding boxes from the image.

        Inputs:
            - df (pd.DataFrame): DataFrame containing bounding box information.
            - img (np.ndarray): Input image.
            - erotion_percent (float): Percentage of erosion for bounding boxes (default: 0).

        Output:
            - img_out (np.ndarray): Image with regions within bounding boxes subtracted.
        '''
    
        img_out = img.copy()
        # img_out = np.array(img)
        for i, row in df.iterrows():

            width = row['right'] - row['left']
            erotion_width = int(round((width * erotion_percent) / 100))

            height = row['bottom'] - row['top']
            erotion_height = int(round((height * erotion_percent) / 100))


            img_out[ (row['top'] + erotion_height)  : (row['bottom'] - erotion_height), 
                     (row['left'] + erotion_width) : (row['right'] - erotion_width) ] = 0

        # await asyncio.sleep(1)

        return img_out
    
    def close_shape_gaps5(self, image, ocr,
                      dist_threshold_percent = 30, 
                      activation_lower_th = 40, 
                      activation_upper_th = 70):
        '''
        Function to close gaps between shapes in an image.

        Inputs:
            - image (np.ndarray): Input processed image.
            - ocr (pd.DataFrame): DataFrame containing OCR information.
            - dist_threshold_percent (int): Percentage threshold for distance threshold (default: 30).
            - activation_lower_th (int): Lower threshold for activation (default: 40).
            - activation_upper_th (int): Upper threshold for activation (default: 70).

        Output:
            - img_out (np.ndarray): Image with gaps closed.
        '''

        img = image.copy()
        img = (1-img) * 10

        kernel = np.ones((3, 3), np.uint8)
        kernel[1,1] = 10

        dst = cv2.filter2D(img,-1,kernel).astype(int)

        points_thr = np.where((dst > activation_lower_th) & (dst < activation_upper_th))

        points = []
        for p_i in range(len(points_thr[0])): 
            points.append([points_thr[0][p_i], points_thr[1][p_i]])

        points = np.stack(points, axis=0)

        nodes_points = []

        nodes_points.extend([[row.top, row.left] for i, row in ocr.iterrows()])
        nodes_points.extend([[row.top, row.right] for i, row in ocr.iterrows()])
        nodes_points.extend([[row.bottom, row.right] for i, row in ocr.iterrows()])
        nodes_points.extend([[row.bottom, row.left] for i, row in ocr.iterrows()])

        nodes_points   = np.array(nodes_points)
        dist_matrix    = euclidean_distances(points)
        max_bb_height  = (ocr.bottom - ocr.top).max()
        dist_threshold = int((max_bb_height * dist_threshold_percent)/100)

        below_th = np.where((dist_matrix < dist_threshold) & (dist_matrix > 0)) # zero is trivial distance, no need to fill any gap

        img_out = image.copy()

        for i in range(len(below_th[0])):

            p1 = points[below_th[0][i]]
            p2 = points[below_th[1][i]]

            dist_to_nodes = euclidean_distances(np.stack([p1, p2]), nodes_points)
            closest_node = np.argmin(dist_to_nodes) % len(ocr)

            closest_node_height = ocr.loc[closest_node, 'bottom'] - ocr.loc[closest_node, 'top']

            dist_threshold = int((closest_node_height * dist_threshold_percent)/100)

            if np.linalg.norm(p2-p1) < dist_threshold:

                cv2.line(img_out, [p1[1],p1[0]], [p2[1],p2[0]],  (1, 1, 1), thickness=1)
        
        # await asyncio.sleep(1)

        return img_out
    
    def stamp_bounding_boxes_on_image(self, df, img, erotion_percent = 10):
        '''
          Function to stamp bounding boxes on an image.

          Inputs:
              - df (pd.DataFrame): DataFrame containing bounding box information.
              - img (np.ndarray): Input image.
              - erotion_percent (int): Percentage of erosion for bounding boxes (default: 10).

          Output:
              - img_out (np.ndarray): Image with bounding boxes stamped.
        '''
    
        img_out = img.copy()
        # img_out = np.array(img)
        for i, row in df.iterrows():

            width = row['right'] - row['left']
            erotion_width = int(round((width * erotion_percent) / 100))

            height = row['bottom'] - row['top']
            erotion_height = int(round((height * erotion_percent) / 100))


            img_out[ (row['top'] + erotion_height)  : (row['bottom'] - erotion_height), 
                     (row['left'] + erotion_width) : (row['right'] - erotion_width) ] = 1


        # await asyncio.sleep(1)

        return img_out
    
    def get_filled_shapes(self,img):
        '''
        Function to extract filled shapes from an image.

        Inputs:
            - img (np.ndarray): Input processed image.

        Output:
            - img_out (np.ndarray): Image with filled shapes.
        '''
    
        contours, tree = cv2.findContours(cv2.convertScaleAbs(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_out = np.zeros_like(img)

        for i, contour in enumerate(contours):
            cv2.drawContours(img_out, [contour], 0, (1, 1, 1), thickness=cv2.FILLED)

        # await asyncio.sleep(1)
        # print('Step 3 complete')

        return img_out
    
    def get_masks(self,img, max_iter=10):
        '''
        Function to obtain masks for nodes and edges.

        Inputs:
            - img (np.ndarray): Input image.
            - max_iter (int): Maximum number of iterations (default: 10).

        Outputs:
            - nodes_mask (np.ndarray): Mask for nodes.
            - edges_mask (np.ndarray): Mask for edges.
        '''
    

        kernel = np.ones((3, 3), np.uint8)

        img_eroded = [img.copy()]
        contours_iter = []

        for i in range(max_iter):
            contours, tree = cv2.findContours(cv2.convertScaleAbs(img_eroded[-1]), 
                                              cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_iter.append(contours)
            img_eroded.append(cv2.erode(img_eroded[-1], kernel, iterations = 1))

        min_contours = len(contours_iter[-1])
        min_contours_iteration = len(contours_iter)-1

        for i in range(len(contours_iter)-1, -1, -1):
            if len(contours_iter[i]) > min_contours:
                min_contours_iteration = i+1
                break


        nodes_mask = img_eroded[min_contours_iteration]

        nodes_mask_dilated = cv2.dilate(nodes_mask, kernel, iterations=min_contours_iteration+1)
        edges_mask = np.maximum((img_eroded[0] - nodes_mask_dilated), 0)

        # await asyncio.sleep(1)

        return nodes_mask, edges_mask
    
    ########################## Edges Direction Processing ###########################


    def preprocess_for_arrow_detection(self,img_gray,dilate,erode):
     
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)

        img_canny = cv2.Canny(img_blur, 50, 50)

        kernel = np.ones((3, 3))

        img_dilate = cv2.dilate(img_canny, kernel, iterations=dilate)
        img_erode  = cv2.erode(img_dilate, kernel, iterations=erode)

        return img_erode
    
    def find_tip(self, points, convex_hull):
    
        length = len(points)

        indices = np.setdiff1d(range(length), convex_hull)

        for i in range(2):

            j = indices[i] + 2
            if j > length - 1:
                j = length - j

            if np.all(points[j] == points[indices[i - 1] - 2]):
                return tuple(points[j])

    def find_arrow_tail(self, arrow_tip, contour):
        # Calculate the distances between the arrow tip and all points in the contour
        distances = [np.linalg.norm(arrow_tip - point[0]) for point in contour]

        # Find the index of the point with the maximum distance (farthest point)
        farthest_point_index = np.argmax(distances)

        # Get the farthest point coordinates
        arrow_tail = tuple(contour[farthest_point_index][0])

        return arrow_tail
    def detect_arrows(self, img, dilate_max=5, erode_max=5, rounding_max = 0.05, rounding_step=0.002):
    
        arrow_contours = []

        arrow_tips = []
        arrow_origins = []
        arrow_lengths = []
        arrow_thickness = []

        dilates = list(range(0, dilate_max))
        erotions = list(range(0, erode_max))
        roundings = np.arange(0.001, rounding_max, rounding_step)

        combinations = []

        for d in dilates:
            for e in erotions:
                for r in roundings:
                    combinations.append({'dilate': d, 'erotion': e, 'rounding': r})

        for comb in combinations:

            contours, hierarchy = cv2.findContours(self.preprocess_for_arrow_detection(img, comb['dilate'], comb['erotion']), 
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, comb['rounding'] * peri, True)
                hull = cv2.convexHull(approx, returnPoints=False)
                sides = len(hull)

                if 6 > sides > 3 and sides + 2 == len(approx):

                    bol_repeated_contour = False

                    for i,c in enumerate(arrow_contours):
                        if cv2.matchShapes(c, cnt, 1, 0.0) < 1:
                            bol_repeated_contour = True
                            break

                    if bol_repeated_contour == False:
                        arrow_contours.append(cnt)
                        arrow_tip = self.find_tip(approx[:, 0, :], hull.squeeze())

                        if arrow_tip:
                            arrow_tips.append(arrow_tip)
                            arrow_tail = self.find_arrow_tail(arrow_tip, cnt)
                            # caculate the lenth
                            length = np.linalg.norm(np.array(arrow_tip) - np.array(arrow_tail))
                            arrow_lengths.append(length)
                            dist_mat = euclidean_distances(np.expand_dims(arrow_tip, axis=0), np.squeeze(cnt))


                            area = cv2.contourArea(cnt)

                            # Calculate the length of the contour
                            contours_length = len(cnt)

                            # Avoid division by zero
                            if contours_length == 0:
                                contours_length = 1

                            # Calculate the thickness of the contour
                            thickness = area / contours_length
                            arrow_thickness.append(thickness)
                            arrow_origin = np.squeeze(cnt)[np.argmax(dist_mat)]
                            arrow_origins.append(arrow_origin)


        arrow_origins = np.array(arrow_origins)
        arrow_tips    = np.array(arrow_tips)
        arrow_lengths = np.array(arrow_lengths)
        arrow_thickness = np.array(arrow_thickness)

        return arrow_origins, arrow_tips,arrow_lengths,arrow_thickness
    
    def get_edges_endpoints_directionality(self,edges_endpoints, tips, origins, dist_threshold=50):

        tips_origins = np.concatenate([tips, origins], axis=1)
        origins_tips = np.concatenate([origins, tips], axis=1)
        # edges_endpoints = np.array(edges_endpoints)
        dist_mat_tips_origins = euclidean_distances(edges_endpoints.reshape((edges_endpoints.shape[0], 4)), tips_origins)
        dist_mat_origins_tips = euclidean_distances(edges_endpoints.reshape((edges_endpoints.shape[0], 4)), origins_tips)

        min_dist = []
        min_dist.append(np.min(dist_mat_tips_origins, axis=1))
        min_dist.append(np.min(dist_mat_origins_tips, axis=1))

        origins_or_tips = np.argmin(min_dist, axis = 0)

        abs_min_dist = np.array([min_dist[selected][i] for i, selected in enumerate(origins_or_tips)])

        directions = [None] * len(edges_endpoints)

        for index in np.where(abs_min_dist < dist_threshold)[0]:
            directions[index] = origins_or_tips[index]

        return directions

    def get_conections(self, nodes_df, edges_endpoints, edges_directionalities, img, dist_threshold_percentage=5):

        nodes_contours = []
        nodes_ids = []
        edge_id = 0
        edges_id = []
        for i, row in nodes_df.iterrows():
            img_out = np.zeros_like(img, dtype=np.uint16)

            img_out[row.top: row.bottom, row.left: row.right] = 1

            contour, tree = cv2.findContours(cv2.convertScaleAbs(img_out), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            assert len(contour) == 1

            nodes_contours.append(contour[0])
            nodes_ids.append(row.node_id)

        edges_endpoints = edges_endpoints.astype(np.uint16)

        connections = []
        destination_nodes = []

        dist_threshold_in_pixels = int((dist_threshold_percentage / 100) * img.shape[0])

        for edge_i, edge in enumerate(edges_endpoints):

            connection = [None, None]

            for i_endpoint, endpoint in enumerate(edge):

                min_dist_to_node = 9e3
                min_dist_node_n = -1

                for i_node, node in enumerate(nodes_contours):

                    min_dist = cv2.pointPolygonTest(node, endpoint, True) * (-1)

                    if min_dist < min_dist_to_node:
                        min_dist_to_node = min_dist
                        min_dist_node_n = nodes_ids[i_node]

                if min_dist_to_node < dist_threshold_in_pixels:
                    connection[i_endpoint] = min_dist_node_n

            if connection[0] is not None and connection[1] is not None and connection[0] != connection[1]:
                connections.append(connection)
                edges_id.append(edge_id)
                if edges_directionalities[edge_i] is not None:
                    dest_node = connection[edges_directionalities[edge_i]]
                else:
                    dest_node = None

                destination_nodes.append(dest_node)
            edge_id = edge_id + 1

        df_pre = pd.DataFrame(connections, columns=['node a', 'node b'])
        df_pre.insert(column='destination node', loc=2, value=destination_nodes)

        #     froze_set = set([frozenset([row['node a'], row['node b']]) for i, row in df_pre.iterrows()])

        #     df = pd.DataFrame(froze_set, columns=['node a', 'node b'])
        #     df['destination node'] = None

        #     for i, row in df.iterrows():

        #         dest_nodes = df_pre.loc[((df_pre['node a'] == row['node a']) & (df_pre['node b'] == row['node b']) |
        #                         (df_pre['node b'] == row['node a']) & (df_pre['node a'] == row['node b'])), 'destination node']

        #         df.at[i, 'destination node'] = dest_nodes.max()
        #     print(edges_id)

        return df_pre, edges_id

    ########################## Edges Direction Processing ###########################

    def get_edges_endpoints(self, edges_mask, min_edge_length_percentage=3):

        final_edges = []

        contour_idswithendpoint = []
        edge_lengths = []

        edge_thickness = []

        min_edge_length_pixels = (min_edge_length_percentage / 100) * edges_mask.shape[0]

        contours, tree = cv2.findContours(cv2.convertScaleAbs(edges_mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_data = []  # A list to store (contour_id, contour, endpoints) tuples
        contour_id = 0

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            edge_lengths.append(perimeter)
            contours_length = len(contour)

            # Avoid division by zero
            if contours_length == 0:
                contours_length = 1
            area = cv2.contourArea(contour)
            # Calculate the thickness of the contour
            thickness = area / contours_length
            edge_thickness.append(thickness)

            c = max([contour], key=cv2.contourArea)

            extreme_points = []

            extreme_points.append(np.array(c[c[:, :, 0].argmin()][0]))
            extreme_points.append(np.array(c[c[:, :, 0].argmax()][0]))
            extreme_points.append(np.array(c[c[:, :, 1].argmin()][0]))
            extreme_points.append(np.array(c[c[:, :, 1].argmax()][0]))

            extreme_points = np.stack(extreme_points, axis=0)

            contour_data.append((contour_id, contour, extreme_points))

            dist_mat = euclidean_distances(extreme_points)
            if np.max(dist_mat) > min_edge_length_pixels:
                ext_indeces = np.unravel_index(np.argmax(dist_mat), shape=dist_mat.shape)

                final_endpoints = [extreme_points[ext_indeces[0]], extreme_points[ext_indeces[1]]]

                final_edges.append(final_endpoints)
                # Step 2 (Continued): Record contour ID of the endpoints

                contour_idswithendpoint.append(contour_id)
            contour_id = contour_id + 1

        n = len(edge_lengths)
        ids = np.arange(n).reshape((-1, 1))
        combined_table = np.column_stack((ids, edge_lengths, edge_thickness))

        selected_rows = combined_table[np.isin(combined_table[:, 0], contour_idswithendpoint)]
        return np.stack(final_edges), selected_rows

    def get_nodes(self, ocr, nodes_mask, threshold_iou=0.8):

        df = ocr.copy()
        nodes_contours, tree = cv2.findContours(cv2.convertScaleAbs(nodes_mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for i, row in df.iterrows():

            area = (row['right'] - row['left']) * (row['bottom'] - row['top'])

            max_iou = 0
            max_iou_i_node = -1

            for i_node, contour in enumerate(nodes_contours):

                empty_img = np.zeros_like(nodes_mask)

                cv2.drawContours(empty_img, [contour], 0, (1, 1, 1), thickness=-1)

                intersection = empty_img[row['top']:row['bottom'], row['left']:row['right']].sum()

                iou = intersection / area

                if iou > max_iou:
                    max_iou = iou
                    max_iou_i_node = i_node

            if max_iou > threshold_iou:
                df.at[i, 'node_id'] = max_iou_i_node

        df['text'] = df.groupby('node_id')['text'].transform(lambda x: '\n'.join(x))
        df.drop_duplicates('text', inplace=True)

        df = df[df.node_id.notna()]
        df.node_id = df.node_id.astype(int)

        df.reset_index(drop=True, inplace=True)

        return df
    def get_datasets(self, annotations_file, images_folder):

        with open(annotations_file) as fp:
            annotations = json.loads(fp.read())

        elements = []

        for i, image_description in enumerate(annotations['images']):

            img = Image.open(os.path.join(images_folder, image_description['file_name']))

            objects = {'id':[], 
                       'area': [],
                       'bbox': [],
                       'category': []}

            for j, ann_description in enumerate(annotations['annotations']):

                if ann_description['image_id'] == image_description['id']:

                    objects['id'].append(ann_description['id'])
                    objects['area'].append(ann_description['area'])
                    objects['bbox'].append(ann_description['bbox'])
                    objects['category'].append(ann_description['category_id'])


            el = {'image_id': image_description['id'],
                  'image': img,
                  'width': image_description['width'],
                  'height': image_description['height'],
                  'objects': objects
            }

            elements.append(el)

        return Dataset.from_list(elements)
    def non_maximum_suppression(self, detections, iou_threshold=0.5):
        """Apply NMS on detections and return filtered detections."""
        detections = sorted(detections, key=lambda x: x['Confidence Score'], reverse=True)
        keep = []
        while detections:
            max_score_det = detections.pop(0)
            keep.append(max_score_det)
            detections = [
                det for det in detections
                if self.compute_iou(max_score_det['Location'], det['Location']) < iou_threshold
            ]
        return keep
    def show_predictions(self, image, threshold=0.9):
    
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = tf.convert_to_tensor([image.size[::-1]], dtype=tf.int32)

        results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

        draw = ImageDraw.Draw(image)

        # List to collect detection details
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            detection_info = {
                'Label': model.config.id2label[label.item()],
                'Confidence Score': round(score.item(), 3),
                'Location': box
            }
            detections.append(detection_info)

            # print(
            #     f"Detected {detection_info['Label']} with confidence "
            #     f"{detection_info['Confidence Score']} at location {detection_info['Location']}"
            # )
            draw.rectangle(box, outline="red", width=1)


        # Convert the list of dictionaries to a DataFrame and save as a CSV file
        detections_df = pd.DataFrame(detections)

        return image, detections_df
    
    #     def get_conections(self, nodes_df, edges_endpoints, img, dist_threshold_percentage=5):

#         nodes_contours = []
#         nodes_ids = []

#         for i, row in nodes_df.iterrows():
#             img_out = np.zeros_like(img, dtype=np.uint16)

#             img_out[row.top: row.bottom, row.left: row.right] = 1

#             contour, tree = cv2.findContours(cv2.convertScaleAbs(img_out), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#             assert len(contour) == 1

#             nodes_contours.append(contour[0])
#             nodes_ids.append(row.node_id)

#         edges_endpoints = edges_endpoints.astype(np.uint16)

#         connections = []

#         dist_threshold_in_pixels = int((dist_threshold_percentage / 100) * img.shape[0])

#         for edge in edges_endpoints:

#             connection = [None, None]

#             for i_endpoint, endpoint in enumerate(edge):

#                 min_dist_to_node = 9e3
#                 min_dist_node_n = -1

#                 for i_node, node in enumerate(nodes_contours):

#                     min_dist = cv2.pointPolygonTest(node, endpoint, True) * (-1)

#                     if min_dist < min_dist_to_node:
#                         min_dist_to_node = min_dist
#                         min_dist_node_n = nodes_ids[i_node]

#                 if min_dist_to_node < dist_threshold_in_pixels:
#                     connection[i_endpoint] = min_dist_node_n

#             if connection[0] is not None and connection[1] is not None and connection[0] != connection[1]:
#                 connections.append(connection)

#         df = pd.DataFrame(connections, columns=['node a', 'node b'])

#         froze_set = set([frozenset([row['node a'], row['node b']]) for i, row in df.iterrows()])

#         df = pd.DataFrame(froze_set, columns=['node a', 'node b'])

#         return df

    def join_close_nodes(self,
                         ocr,
                         vertical_distance_threshold_percent=25,
                         horizontal_distance_threshold_percent=50):

        df = ocr.copy()

        while True:

            to_add = []
            to_remove = []

            flag_updates_made = False

            for i,row_a in df.iterrows():
                for j,row_b in df.iterrows():

                    row_a_height = row_a.bottom - row_a.top
                    row_b_height = row_b.bottom - row_b.top

                    row_a_width = row_a.right - row_a.left
                    row_b_width = row_b.right - row_b.left

                    mean_height = (row_a_height + row_b_height) / 2
                    mean_width  = (row_a_width + row_b_width) / 2

                    vertical_distance_threshold_pixels   = (vertical_distance_threshold_percent / 100) * mean_height
                    horizontal_distance_threshold_pixels = (horizontal_distance_threshold_percent / 100) * mean_width

                    if (j > i and 
                        abs(row_b.top - row_a.bottom) < vertical_distance_threshold_pixels and
                        abs(row_b.left - row_a.left) < horizontal_distance_threshold_pixels):

                        df.at[i, 'text'] = row_a.text + ' ' + row_b.text
                        df.at[i, 'bottom'] = row_b.bottom
                        df.at[i, 'left'] = min(row_a.left, row_b.left)
                        df.at[i, 'right'] = max(row_a.right, row_b.right)
                        df.at[i, 'font_size'] = (row_a.font_size + row_b.font_size) / 2

                        df = df.drop(j, axis=0)
                        df.reset_index(drop=True, inplace=True)

                        flag_updates_made = True

                        break

                if flag_updates_made:
                    break

            if flag_updates_made == False:
                break

        return df
    
    
    def spellcheck2(self,text):
        parser = GingerIt()
        text = text.replace('&', 'and')
        res = parser.parse(text)
        output = res['result']
        output = output.replace(' and ', ' & ')
        return output
    
    def colordetect(self,hue_value):
        '''
        Function to detect the color based on the given hue value.

        Inputs:
            - hue_value (int): Hue value.

        Output:
            - color (str): Detected color.
        '''
        color_ranges = {
            (0, 5): 'RED',
            (5, 22): 'ORANGE',
            (22, 33): 'YELLOW',
            (33, 78): 'GREEN',
            (78, 151): 'BLUE',
            (151, 167): 'VIOLET',
            (167, 200): 'PINK',  
            (200, 240): 'PURPLE',  
            (240, 300): 'CYAN', 
            (300, 360): 'MAGENTA'  
        }

        if hue_value == 0:
            return 'BLACK'

        for range_min, range_max in color_ranges:
            if hue_value >= range_min and hue_value < range_max:
                return color_ranges[(range_min, range_max)]

        return 'UNKNOWN'
    
    def nodes_addcolor(self,nodes_df, image_file):
        '''
        Function to add color information to the nodes DataFrame.

        Inputs:
            - nodes_df (pd.DataFrame): DataFrame containing node information.
            - image_file (str): Path to the image file.

        Output:
            - nodes_df (pd.DataFrame): Updated DataFrame with color information.
        '''
        
        # image = cv2.imread(image_file)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = self.open_image(image_file)
        for index, row in nodes_df.iterrows():
            left = int(row["left"])
            top = int(row["top"])
            right = int(row["right"])
            bottom = int(row["bottom"])

            pixels = []
            non_zero_hue_found = False  # set boolean value

            for _ in range((right - left + 1) * (bottom - top + 1) // 2):
                x = random.randint(left, right)
                y = random.randint(top, bottom)
                color = tuple(image[y, x])
                hue_value = color[0]

                if hue_value != 0:
                    pixels.append(color)
                    non_zero_hue_found = True  # set True，because we find the hue_value not equal zero

            if non_zero_hue_found:
                color_counter = Counter(pixels)
                most_common_color = color_counter.most_common(1)[0][0]
            else:
                most_common_color = (0, 0, 0)  # hue = 0

            # node_id = float(row["node_id"])
            nodes_df.at[index, "color"] = self.colordetect(most_common_color[0])
            # data.at[index, "node_id"] = node_id
        # await asyncio.sleep(1)
        
        return nodes_df
    
    def compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2

        # Calculate intersection rectangle coordinates
        xA = max(x1, x1_)
        yA = max(y1, y1_)
        xB = min(x2, x2_)
        yB = min(y2, y2_)

        # Calculate the area of intersection
        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Calculate the area of each box
        box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)

        # Calculate the area of union
        union_area = box1_area + box2_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area

        return iou

    def extract_feature(self,json_file,image_file):
        '''
        Function to extract features from the OCR data and image.

        Inputs:
            - json_file (str): Path to the JSON file containing OCR data.
            - image_file (str): Path to the image file.

        Outputs:
            - nodes_df (pd.DataFrame): DataFrame containing node information.
            - connections_df (pd.DataFrame): DataFrame containing connection information.
        '''
        image_open = Image.open(os.path.join(image_file))
        print('Step 1: use the node model to predict the node.')
        image_test,df_test =  self.get_node_predictions(image_open, 0.7)
        print('Step 2: Start to read the ocr file.')
        ocr = self.get_ocr_data(json_file)
        imagetest = []
        # image_file = Image.fromarray(image_file)
        imagetest.append(image_open)
        df_ocr_test = self.set_bounding_boxes_in_pixels(ocr, imagetest[-1])
        df_ocr_test = self.get_font_size(df_ocr_test)
        df_ocr_test = self.join_close_nodes(df_ocr_test, vertical_distance_threshold_percent=25, horizontal_distance_threshold_percent=50)
        df_ocr_test['Location'] = df_ocr_test.apply(lambda row: [row['left'], row['top'], row['right'], row['bottom']], axis=1)
        df_ocr_test = df_ocr_test[['text', 'font_size', 'Location']]
        threshold = 0.3
        matched_rows = []

        for i, row in df_test.iterrows():
            for _, row_ocr in df_ocr_test.iterrows():
                iou = self.compute_iou(row['Location'], row_ocr['Location'])
                if iou >= threshold:
                    matched_rows.append(i)
                    break
        matched_row_ids = matched_rows
        # 1. Identify unmatched rows
        unmatched_df = df_test.loc[~df_test.index.isin(matched_row_ids)].reset_index(drop=True)

        # 2. Create a new dataframe and adjust columns
        unmatched_df = unmatched_df[['Location']]
        unmatched_df.columns = ['Location']

        # 3. Add 'text' column
        unmatched_df['text'] = ['drawing_' + str(i) for i in range(unmatched_df.shape[0])]

        # 4. Append to df_ocr_test
        df_ocr_test_final = pd.concat([df_ocr_test, unmatched_df[['text', 'Location']]], ignore_index=True)
        
        df_ocr_test_final['left'] = df_ocr_test_final['Location'].apply(lambda x: x[0])
        df_ocr_test_final['top'] = df_ocr_test_final['Location'].apply(lambda x: x[1])
        df_ocr_test_final['right'] = df_ocr_test_final['Location'].apply(lambda x: x[2])
        df_ocr_test_final['bottom'] = df_ocr_test_final['Location'].apply(lambda x: x[3])

        # Drop the original 'Location' column if needed
        df_ocr_test_final = df_ocr_test_final.drop(columns=['Location'])

        print('Step 3: Start to use drawing model to predict.')
        
        drawing_test,drawing_df_test =  self.get_drawings_predictions(image_open, 0.7)
        df_ocr_test_final['Location'] = df_ocr_test_final.apply(lambda row: [row['left'], row['top'], row['right'], row['bottom']], axis=1)

        df_ocr_final = df_ocr_test_final.drop(columns=['left', 'top', 'right', 'bottom'])
        final_threshold = 0.3
        final_matched_rows = []

        for i, row in drawing_df_test.iterrows():
            for _, row_ocr in df_ocr_final.iterrows():
                iou = self.compute_iou(row['Location'], row_ocr['Location'])
                if iou >= threshold:
                    final_matched_rows.append(i)
                    break
        unmatched_df = drawing_df_test.loc[~drawing_df_test.index.isin(final_matched_rows)].reset_index(drop=True)

        unmatched_df = unmatched_df[['Location']]
        unmatched_df.columns = ['Location']

        unmatched_df['text'] = ['drawing_0' + str(i) for i in range(unmatched_df.shape[0])]

        final_node_df = pd.concat([df_ocr_final, unmatched_df[['text', 'Location']]], ignore_index=True)
        print('Step 4: Start to use edges model to predict.')
        all_nodes_df = final_node_df
        all_nodes_df['left'] = all_nodes_df['Location'].apply(lambda x: x[0])
        all_nodes_df['top'] = all_nodes_df['Location'].apply(lambda x: x[1])
        all_nodes_df['right'] = all_nodes_df['Location'].apply(lambda x: x[2])
        all_nodes_df['bottom'] = all_nodes_df['Location'].apply(lambda x: x[3])
        # Drop the original 'Location' column if needed
        all_nodes_df = all_nodes_df.drop(columns=['Location'])
        all_nodes_df_nofont = all_nodes_df.drop(columns=['font_size'])
        all_nodes_df_nofont['top'] = all_nodes_df_nofont['top'].astype(int)
        all_nodes_df_nofont['right'] = all_nodes_df_nofont['right'].astype(int)
        all_nodes_df_nofont['left'] = all_nodes_df_nofont['left'].astype(int)
        all_nodes_df_nofont['bottom'] = all_nodes_df_nofont['bottom'].astype(int)
        
        # print('Step 4: Start to use edges model to predict.')
        read_image = self.open_image(image_file)
        image_final_test = []
        image_final_test.append(self.open_image(image_file))
        image_final_test.append(self.stamp_bounding_boxes_on_image(all_nodes_df_nofont, image_final_test[-1]))
        all_nodes_df_withfont = self.get_font_size(all_nodes_df_nofont)
        image_no_thresholding = cv2.convertScaleAbs(self.substract_bounding_boxes(all_nodes_df_withfont, read_image, 20))
        # print(image_no_thresholding)
        arrow_origins, arrow_tips, arrow_length,arrow_thickness = self.detect_arrows(img=image_no_thresholding, dilate_max=5, erode_max=5, rounding_max = 0.05, rounding_step=0.002)
        image_final_test.append(self.threshold_image(image_final_test[-1]))
        image_final_test.append(self.stamp_bounding_boxes_on_image(all_nodes_df_withfont, image_final_test[-1], erotion_percent = 20))

#         threshold_image= self.threshold_image(read_image, threshold_value=200)

#         ocr = self.set_bounding_boxes_in_pixels(ocr, threshold_image)
#         ocr = self.get_font_size(ocr)
#         ocr = self.join_close_nodes(ocr, 
#                                     vertical_distance_threshold_percent=25, 
#                                     horizontal_distance_threshold_percent=50)

        
        #image.append(substract_bounding_boxes(ocr, image[-1], erotion_percent = 0))
        # image_with_stamp_bounding_box = self.stamp_bounding_boxes_on_image(ocr,
        #                                                                    threshold_image,
        #                                                                    erotion_percent = 10)
        
        # close_shape_image = self.close_shape_gaps5(image_with_stamp_bounding_box, 
        #                                             ocr, 
        #                                             dist_threshold_percent = 50)

#         filled_shapes_image = self.get_filled_shapes(image_with_stamp_bounding_box)
        print('Step 5: Start to compute nodes and edges masks.')
        nodes_mask, edges_mask = self.get_masks(image_final_test[-1], max_iter=5)
        image_final_test.append(nodes_mask)
        image_final_test.append(edges_mask)

        edges_endpoints, edges_endpoints_features = self.get_edges_endpoints(edges_mask, min_edge_length_percentage=1.5)

        edges_directionalities = self.get_edges_endpoints_directionality(edges_endpoints, arrow_tips, arrow_origins, dist_threshold=50)
        
        #uses vit to predict edge_tip and compares both opencv and vit approaches using a simple logic
        final_edges_directionalities = self.get_edges_endpoints_directionality_vit(edges_endpoints, image_final_test[0], 
                                                                              edges_directionalities, score_threshold=0.3)


        nodes_df = self.get_nodes(all_nodes_df_withfont, nodes_mask, threshold_iou = 0.3)

        nodes_df['text'] = nodes_df.text.apply(self.spellcheck2)
        nodes_df = self.nodes_addcolor(nodes_df, image_file)
        connections_df,edges_id = self.get_conections(nodes_df, edges_endpoints, final_edges_directionalities, 
                                        image_final_test[-1], dist_threshold_percentage = 20)


        final_edges_feature = edges_endpoints_features[edges_id]
        df_edge_features = pd.DataFrame(final_edges_feature, columns=['id', 'length', 'thickness'])

        connections_df = pd.concat([connections_df, df_edge_features], axis=1)

        connections_df = self.remove_repeated_connections(connections_df)
        
#         print('Step 6: Start to extract nodes features')
#         nodes_df = self.get_nodes(ocr,nodes_mask,threshold_iou = 0.3)
#         # check the text spelling
#         nodes_df['text'] = nodes_df.text.apply(self.spellcheck2)
#         # add the colour label for nodes

        
#         print('Step 6: Start to extract edges features')
#         ## determin edge with arrow origins and arrow tips
#         image_no_thresholding = cv2.convertScaleAbs(self.substract_bounding_boxes(ocr, read_image, 20))
#         arrow_origins, arrow_tips = self.detect_arrows(image_no_thresholding, 
#                                                        dilate_max=5, 
#                                                        erode_max=5, 
#                                                        rounding_max = 0.05, 
#                                                        rounding_step=0.002)
        
#         # determind the edge directionalties
#         edges_endpoints, edges_endpoints_features = self.get_edges_endpoints(edges_mask, min_edge_length_percentage=1.5)
#         edges_directionalities = self.get_edges_endpoints_directionality(edges_endpoints, 
#                                                                          arrow_tips, 
#                                                                          arrow_origins, dist_threshold=50)

#         # uses vit to predict edge_tip and compares both opencv and vit approaches using a simple logic
#         final_edges_directionalities = self.get_edges_endpoints_directionality_vit(edges_endpoints, read_image,
#                                                                               edges_directionalities,
#                                                                               score_threshold=0.3)

#         # final_edges_feature = edges_endpoints_features[edges_id]
#         # df_edge_features = pd.DataFrame(final_edges_feature, columns=['id', 'length', 'thickness'])

#         connections_df,edges_id = self.get_conections(nodes_df, 
#                                              edges_endpoints, 
#                                              final_edges_directionalities, 
#                                              edges_mask, 
#                                              dist_threshold_percentage = 20)
        
# #         connections_df = pd.concat([connections_df, df_edge_features], axis=1)

# #         connections_df = self.remove_repeated_connections(connections_df)
        

#         final_edges_feature = edges_endpoints_features[edges_id]

#         df_edge_features = pd.DataFrame(final_edges_feature, columns=['id', 'length', 'thickness'])[["length","thickness"]]
#         connections_df = pd.concat([connections_df, df_edge_features], axis=1)

        node_a_counts = nodes_df['node_id'].map(connections_df['node a'].value_counts())

        node_b_counts = nodes_df['node_id'].map(connections_df['node b'].value_counts())

        nodes_df['total_counts'] = node_a_counts.add(node_b_counts, fill_value=0)
        nodes_df['node_level'] = np.where(nodes_df['total_counts'] >= 3, 1, np.where(nodes_df['total_counts'] >= 2, 2, 3))
        nodes_df.drop('total_counts', axis=1)
        
        return nodes_df, connections_df

    def generate_json_output(self,nodes_df,connections_df,image_file,user_id):

        '''
        Function to generate the JSON output file.

        Inputs:
            - nodes_df (pd.DataFrame): DataFrame containing node information.
            - connections_df (pd.DataFrame): DataFrame containing connection information.
            - image_file (str): Path to the image file.

        Output:
            - graph (dict): Graph dictionary containing nodes, edges, and image information.
        '''
        # Create nodes dictionary
        image_id = str(uuid.uuid4())
        nodes = []
        for index, row in nodes_df.iterrows():
            node_id = row['node_id']
            attributes = {
                'position':{'x': (row['left'] + row['right']) / 2,  
                'y': (row['top'] + row['bottom']) / 2},

                'color': row['color'],
                'font_size': row['font_size'],
                'text': row['text'],
                'node_level': row['node_level']
            }
            node = {
                'node_id': node_id,
                'attributes': attributes
            }
            nodes.append(node)

        # Create edges list
        edges = []
        for index, row in connections_df.iterrows():
            edge = {
                'node a': int(row['node a']),
                'node b': int(row['node b']),
              }
            if pd.isna(row['destination node']):
                destination_node = 'None'
            else:
                destination_node = int(row['destination node'])
            
            edge_info={
                'edge':edge,
                'destination_node':destination_node,
                'thickness': float(row['thickness']),
                'length': float(row['length'])
            } 
            edges.append(edge_info)

        # Create graph dictionary
        graph = {'image_id':image_id,
                 'user_id': user_id,
                 'image_name':image_file,
                 'graph':{'nodes': nodes,
                        'edges': edges
                    }
                }
        # output_file = image_file.replace('jpg','json')
        # Write graph to JSON file
        # with open(output_file, 'w') as fp:
        #     json.dump(graph, fp, indent=4)
        
        return graph

##########################  Load detectors (ViT) models ###########################



    def get_drawings_predictions(self, image, threshold=0.1):

        # img = Image.fromarray(image)
#         inputs = self.drawing_image_processor(images=img, return_tensors="pt")
#         outputs = self.drawing_model(**inputs)

#         target_sizes = torch.tensor([img.size[::-1]])
#         return \
#         self.drawing_image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[
#             0]
        inputs = self.drawing_image_processor(images=image, return_tensors="pt")
        outputs = self.drawing_model(**inputs)

        target_sizes = tf.convert_to_tensor([image.size[::-1]], dtype=tf.int32)

        results = self.drawing_image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

        draw = ImageDraw.Draw(image)

        # List to collect detection details
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            detection_info = {
                'Label': self.drawing_model.config.id2label[label.item()],
                'Confidence Score': round(score.item(), 3),
                'Location': box
            }
            detections.append(detection_info)

            # print(
            #     f"Detected {detection_info['Label']} with confidence "
            #     f"{detection_info['Confidence Score']} at location {detection_info['Location']}"
            # )
            draw.rectangle(box, outline="red", width=1)


        # Convert the list of dictionaries to a DataFrame and save as a CSV file
        detections_df = pd.DataFrame(detections)

        return image, detections_df

    def get_edge_tip_predictions(self, image, threshold=0.1):

        img = Image.fromarray(image)
        inputs = self.edge_tip_image_processor(images=img, return_tensors="pt")
        outputs = self.edge_tip_model(**inputs)

        target_sizes = tf.convert_to_tensor([image.size[::-1]], dtype=tf.int32)

        return \
        self.edge_tip_image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[
            0]

    def get_node_predictions(self, image, threshold=0.7):

#         # img = Image.fromarray(image)
#         inputs = self.node_image_processor(images=img, return_tensors="pt")
#         outputs = self.node_model(**inputs)

#         target_sizes = torch.tensor([img.size[::-1]])
#         return \
#         self.node_image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
        inputs = self.node_image_processor(images=image, return_tensors="pt")
        outputs = self.node_model(**inputs)

        target_sizes = tf.convert_to_tensor([image.size[::-1]], dtype=tf.int32)

        results = self.node_image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

        draw = ImageDraw.Draw(image)

        # List to collect detection details
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            detection_info = {
                'Label': self.node_model.config.id2label[label.item()],
                'Confidence Score': round(score.item(), 3),
                'Location': box
            }
            detections.append(detection_info)

            # print(
            #     f"Detected {detection_info['Label']} with confidence "
            #     f"{detection_info['Confidence Score']} at location {detection_info['Location']}"
            # )
            draw.rectangle(box, outline="red", width=1)


        # Convert the list of dictionaries to a DataFrame and save as a CSV file
        detections_df = pd.DataFrame(detections)

        return image, detections_df

    def get_edges_endpoints_directionality_vit(self, edges_endpoints, image, edges_directionalities, score_threshold):

        edge_tips_vit = self.get_edge_tip_predictions(image)

        tolerance_margin_pixels = 5
        min_score = 0.3
        min_difference_score = 0.1

        edges_directionalities_vit = []
        vit_scores = []

        for i, edge_endpoint in enumerate(edges_endpoints):

            x_0 = edge_endpoint[0][0]
            y_0 = edge_endpoint[0][1]

            x_1 = edge_endpoint[1][0]
            y_1 = edge_endpoint[1][1]

            scores_0 = [0]
            scores_1 = [0]

            for j, edge_tip_vit in enumerate(edge_tips_vit['boxes']):

                bb_x0 = edge_tip_vit[0] - tolerance_margin_pixels
                bb_x1 = edge_tip_vit[2] + tolerance_margin_pixels

                bb_y0 = edge_tip_vit[1] - tolerance_margin_pixels
                bb_y1 = edge_tip_vit[3] + tolerance_margin_pixels

                if (x_0 >= bb_x0 and x_0 <= bb_x1 and
                        y_0 >= bb_y0 and y_0 <= bb_y1):
                    scores_0.append(round(float(edge_tips_vit['scores'][j]), 2))

                if (x_1 >= bb_x0 and x_1 <= bb_x1 and
                        y_1 >= bb_y0 and y_1 <= bb_y1):
                    scores_1.append(round(float(edge_tips_vit['scores'][j]), 2))

            max_score_0 = np.max(scores_0)
            max_score_1 = np.max(scores_1)

            if max_score_0 > min_score and max_score_0 > max_score_1 + min_difference_score:
                edges_directionalities_vit.append(0)
                vit_scores.append(max_score_0)

            elif max_score_1 > min_score and max_score_1 > max_score_0 + min_difference_score:
                edges_directionalities_vit.append(1)
                vit_scores.append(max_score_1)

            else:
                edges_directionalities_vit.append(None)
                vit_scores.append(0)

        # compare vit and opencv approaches

        final_edges_directionalities = []

        for i in range(len(edges_directionalities)):

            if edges_directionalities[i] == edges_directionalities_vit[i]:

                final_edges_directionalities.append(edges_directionalities[i])



            elif edges_directionalities[i] == None and vit_scores[i] > score_threshold:

                final_edges_directionalities.append(edges_directionalities_vit[i])



            elif edges_directionalities_vit[i] == None:

                final_edges_directionalities.append(edges_directionalities[i])


            else:

                if vit_scores[i] > score_threshold:
                    final_edges_directionalities.append(edges_directionalities_vit[i])
                else:
                    final_edges_directionalities.append(edges_directionalities[i])

        return final_edges_directionalities

    def remove_repeated_connections(self, df):

        to_remove = []

        for i, row in df.iterrows():

            repeated_rows = df[(((df['node a'] == row['node a']) & (df['node b'] == row['node b'])) |
                                ((df['node a'] == row['node b']) & (df['node b'] == row['node a'])))]

            if len(repeated_rows) > 1:

                flag_done = False

                for j, row2 in repeated_rows.iterrows():

                    if np.isnan(row2['destination node']) == False:
                        to_remove.extend(repeated_rows.index)
                        to_remove.remove(j)
                        flag_done = True
                        break

                if flag_done == False:
                    to_remove.extend(repeated_rows.index[1:])

        df.drop(index=list(set(to_remove)), inplace=True)

        return df