#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os, cv2
import numpy as np


# In[ ]:


def get_structure_properties(shapes):
    x= shapes['all_points_x']
    y= shapes['all_points_y']
    points= []
    contour= []
    for i, val in enumerate(x):
        points.append(val)
        points.append(y[i])
        contour.append([val, y[i]])
    ctr= np.array(contour).reshape((-1,1,2)).astype(np.int32)
    area= cv2.contourArea(ctr)
    rect= cv2.boundingRect(ctr)
    x,y,w,h = rect
    bbox= [x, y, w, h]
    
    return points, bbox, area


# In[ ]:


def via_to_coco(infile, outfile, image_path):

    vgg_json= open(infile)
    vgg_json= json.load(vgg_json)
    
    main_dict= {}
    info= "{'year': 2020, 'version': '1', 'description': 'Exported using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)', 'contributor': '', 'url': 'http://www.robots.ox.ac.uk/~vgg/software/via/', 'date_created': 'Sun Feb 02 2020 11:47:26 GMT+0100 (Central European Standard Time)'}"
    image_list= list(vgg_json.keys())
    
    images= []
    
    for i, img in enumerate(image_list):
        image= {}
        im= cv2.imread(image_path+img)
        h, w, c= im.shape
        image['id']= i
        image['width']= w
        image['height']= h
        image['file_name']= str(img)
        image['coco_url']= str(img)
        images.append(image)

    annotations= []
    image_id= 0

    for i, v in enumerate(vgg_json):
        data= vgg_json[v]
        regions= data['regions']
        for j, r in enumerate(regions):
            shape_attributes= r['shape_attributes']
            region_attributes= r['region_attributes']
            try:
                # replace the key Objekte with yours
                objekt= region_attributes['Objekte']
            except:
                print('No Object keyword for ', v)
                continue
            segmentation, bbox, area= get_structure_properties(shape_attributes)
            anno= {}
            anno['id']= image_id
            anno['image_id']= i
            anno['segmentation']= segmentation
            anno['area']= area
            anno['bbox']= bbox
            anno['iscrowd']= 0
            image_id+= 1
            annotations.append(anno)

    main_dict['info']= info
    main_dict['images']= images
    main_dict['annotations']= annotations
    with open(outfile, 'w') as f:
        json.dump(main_dict, f)
        f.close()


# In[ ]:


if __name__ == '__main__':
    via_to_coco('path_to_input_via_json_file', 'path_to_output_coco_json_file', 'path_to_images')

