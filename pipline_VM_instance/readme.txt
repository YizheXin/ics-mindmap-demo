Step 1: Start to Read the OCR Text 
use OCR to extract text information from the mind map, it returns the text information, font size, coordinates.

and we use features to determine the nodes destination.

step 2: processing the image
image_with_stamp_bounding_box = pipeline.stamp_bounding_boxes_on_image(ocr,threshold_image,erotion_percent = 10)    

this method will create the stamp of bonding box, and the erotion percent represent the size of the stamp, we do this  in order to separate the nodes and edges better. 
filled_shapes_image = pipeline.get_filled_shapes(image_with_stamp_bounding_box)
this method will fill the closest shape for the stamp of bonding box, in order to reduce the noise for edges
step 3: get nodes mask and edges mask
last step we get the stamps, that things can be seen as nodes mask, and we remove the nodes mask we get the edges mask.
step 4: Start to Extract Nodes Features

pipeline.nodes_addcolor(nodes_df, image_file)
this method will detect the color of the nodes. 



step 5: Start to Extract edges Features


arrow_origins, arrow_tips = pipeline.detect_arrows(image_no_thresholding, dilate_max=5, erode_max=5, rounding_max = 0.05, rounding_step=0.002)
here is three parameters useful for the final performance:
1- numbers of iteration for dilate
2- numbers of iteration for erode
3- a parameter 'rounding_max ' that controls the precision of polygon approximation.

4- a parameter 'rounding_step' that represent the number adding in one step
step 6: determine the node level
nodes_df['node_level'] = np.where(nodes_df['total_counts'] >= 3, 1, np.where(nodes_df['total_counts'] >= 2, 2, 3))
based on the connections of a node to determine level


step 7: generate json output format as prompts to LLM