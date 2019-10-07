product_details={
    1: {'id': 1, 'name': 'coke-1.5l', 'price': 72},
    2: {'id': 2, 'name': 'coke-700ml', 'price': 79},
    3: {'id': 3, 'name': 'coke-bottle-2l', 'price': 89},
    4: {'id': 4, 'name': 'coke-bottle-2l-zero', 'price': 101},
    5: {'id': 5, 'name': 'coke-bottle-2ll', 'price': 120},
    6: {'id': 6, 'name': 'coke-bottle-300ml', 'price': 30},
    7: {'id': 7, 'name': 'coke-bottle-700ml', 'price': 40},
    8: {'id': 8, 'name': 'coke-can', 'price': 15},
    9: {'id': 9, 'name': 'coke-can-classic', 'price': 10},
    10: {'id': 10, 'name': 'coke-can-diet', 'price': 25},
    11: {'id': 11, 'name': 'coke-can-energy', 'price': 25},
    12: {'id': 12, 'name': 'coke-can-espresso', 'price': 26},
    13: {'id': 13, 'name': 'coke-can-orange', 'price': 20},
    14: {'id': 14, 'name': 'coke-can-zero', 'price': 29},
    15: {'id': 15, 'name': 'coke-glass', 'price': 10},
    16: {'id': 16, 'name': 'coke-glass-diet', 'price': 15},
    17: {'id': 17, 'name': 'coke-light-bottle-300ml', 'price': 30},
    18: {'id': 18, 'name': 'coke-zero-bottle-2l', 'price': 101},
    19: {'id': 19, 'name': 'coke-zero-bottle-700ml', 'price': 40},
    20: {'id': 20, 'name': 'gooday-large', 'price': 150},
    21: {'id': 21, 'name': 'gooday-medium', 'price': 60},
    22: {'id': 22, 'name': 'gooday-mediuml-choco', 'price': 70},
    23: {'id': 23, 'name': 'gooday-small', 'price': 20},
    24: {'id': 24, 'name': 'gooday-small-5', 'price': 5},
    25: {'id': 25, 'name': 'gooday-small-butter', 'price': 10},
    26: {'id': 26, 'name': 'gooday-small-choco', 'price': 10},
    27: {'id': 27, 'name': 'lays-large', 'price': 40},
    28: {'id': 28, 'name': 'lays-med', 'price': 20},
    29: {'id': 29, 'name': 'lays-small', 'price': 10},
    30: {'id': 30, 'name': 'lays-small-5', 'price': 5},
    31: {'id': 31, 'name': 'pepsi-can', 'price': 15},
    32: {'id': 32, 'name': 'pepsi-can-0', 'price': 20},
    33: {'id': 33, 'name': 'pepsi-can-pack-2', 'price': 40},
    34: {'id': 34, 'name': 'sprite-can', 'price': 15},
    35: {'id': 35, 'name': 'thumbsup-glass', 'price': 10}
}


detection_graph=''
label_map=''
categories=''
category_index=''
sess=''

def gvs(PATH_TO_IMAGE):
    ######## Image Object Detection Using Tensorflow-trained Classifier #########
    #
    # Author: Evan Juras
    # Date: 1/15/18
    # Description:
    # This program uses a TensorFlow-trained classifier to perform object detection.
    # It loads the classifier uses it to perform object detection on an image.
    # It draws boxes and scores around the objects of interest in the image.

    ## Some of the code is copied from Google's example at
    ## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

    ## and some is copied from Dat Tran's example at
    ## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

    ## but I changed it to make it more understandable to me.

    # Import packages
    import os
    import cv2
    import numpy as np
    import tensorflow as tf
    import sys

    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")

    # Import utilites
    from utils import label_map_util
    from utils import visualization_utils as vis_util

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'
    # IMAGE_NAME = 'test12.jpg'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph20000.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap_collab4_10.pbtxt')

    # Path to image
    # PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

    # Number of classes the object detector can identify
    NUM_CLASSES = 35

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    global category_index
    global label_map
    global categories
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    global detection_graph
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        global sess
        sess = tf.Session(graph=detection_graph)
    return gvs2(PATH_TO_IMAGE)

def gvs2(PATH_TO_IMAGE):
    # Import packages
    import os
    import cv2
    import numpy as np
    import sys
    # Import utilites
    from utils import label_map_util
    from utils import visualization_utils as vis_util

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    #gvs works here

    # print("category_index")
    # print(category_index)
    # print(type(category_index))
    # print(category_index[22])
    # print("category_index")

    # print("classes")
    # print(detection_classes)
    # print(type(detection_classes))
    # print(list(classes[0]))
    # print(type(classes[0]))
    # print("classes")

    # print("scores")
    # print(scores)
    # print(type(scores))
    # print(list(scores[0]))
    # print(type(scores[0]))
    # print("scores")

    #gvs works here


    import csv
    final_score = np.squeeze(scores)
    count = 0
    for i in range(100):
        if scores is None or final_score[i] > 0.6:
                count = count + 1
    print('Detected Product :',count)
    printcount =0;
    lister =[]
    if count!=0:
        for i in classes[0]:
            if(printcount == count):
                break
            printcount = printcount +1
            with open('product_list.csv', mode='a',newline='') as product_file:
                product_writer = csv.writer(product_file)
                lister.append(category_index[i]['id'])
                product_writer.writerow(lister)
        return 1
    else:
        return PATH_TO_IMAGE



    # Draw the results of the detection (aka 'visulaize the results')

    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=3,
    #     min_score_thresh=0.60)

    # # All the results have been drawn on image. Now display the image.
    # cv2.imshow('Object detector', image)

    # # Press any key to close the image
    # cv2.waitKey(0)

    # # Clean up
    # cv2.destroyAllWindows()
    # print(lister)
