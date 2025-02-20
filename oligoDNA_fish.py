import os
import cv2
import csv
import tifffile
import numpy as np
from tqdm import tqdm
from skimage import measure
from cellpose import models, utils

import warnings
warnings.filterwarnings("ignore") # Suppress warnings for cleaner output


def mask_cells(output_path, fname, cells_im, avg_cells_size, cellpose_model_path):
    """
    Segments the cells in a 3D image and saves the mask in a "cells_mask" subfolder.

    Parameters:
        output_path (str): Path where the results will be saved.
        fname (str): Name of the file being processed.
        cells_im (numpy.ndarray): 3D image of cells (normalized).
        avg_cells_size (float): Average cell size for segmentation.
        cellpose_model_path (str or None): Path to a pretrained Cellpose model. If None, uses 'cyto3' model.

    Returns:
        numpy.ndarray: 3D mask of segmented cells.
    """
    mask_path = os.path.join(output_path, 'cells_mask', fname+'_mask.tif')
    
    # Check if the mask already exists, otherwise generate it
    if not os.path.exists(mask_path):
        print('# Generating cells mask')
        
        # Use default Cellpose 'cyto3' model if no custom model is provided
        if cellpose_model_path is None:
            print('## Using cyto3 model')
            cellpose_model = models.Cellpose(gpu=True, model_type='cyto3')
        else:
            print('## Using retrained model')
            cellpose_model = models.CellposeModel(gpu=True, pretrained_model=cellpose_model_path)
        
        # Perform cell segmentation
        mask, _, _, _ = cellpose_model.eval(cells_im, channels=[0,0], diameter=avg_cells_size, stitch_threshold=0.5)
        
        # Create the output directory if it does not exist
        os.makedirs(os.path.join(output_path, 'cells_mask'), exist_ok=True)
        
        # Save the mask as a TIFF file
        tifffile.imwrite(mask_path, mask, imagej=True)
        
        return mask
    
    # If the mask exists, load and return it
    return tifffile.imread(mask_path)


def mask_spots(spots_im, output_path, fname, locus_id):
    """
    Segments fluorescent spots (locus) in a 3D image and saves the mask.

    Parameters:
        spots_im (numpy.ndarray): 3D image of spots.
        output_path (str): Path where results will be saved.
        fname (str): Name of the file being processed.
        locus_id (str): Identifier for the locus layer ('l1' or 'l2').

    Returns:
        numpy.ndarray: 3D mask of the segmented spots.
    """
    print(f'# Generating {locus_id} mask')
    locus_mask = np.zeros(spots_im.shape, dtype=np.uint8)
    
    for z in range(spots_im.shape[0]):
        # Apply a small blur to reduce noise
        blur_im = cv2.blur(spots_im[z].copy(), (2,2))
        
        # Normalize the image intensity between 0 and 255
        norm_im = cv2.normalize(blur_im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # Segment the image thresholding it
        """
        IF YOU WANT TO CHANGE THRESHOLD VALUE TO IMPROVE THE RESULTS, HERE IS THE PLACE!
        1. change the X value in the following row:
                _, spots_mask = cv2.threshold(norm_im, X, 255, cv2.THRESH_BINARY)
           For us, by default X = 60
        2. to visualize the new threshold mask and check the new X value works, you can use:
                plt.imshow(spots_im[z])
                plt.imshow(spots_mask, alpha=0.4)
                plt.show()
            copy these lines just after the thresholding line above, this will
            open you a window with the z section image and with the threshold mask
            drawn on it
        """
        # Apply thresholding to segment the spots
        _, spots_mask = cv2.threshold(norm_im, 60, 255, cv2.THRESH_BINARY)
        spots_mask = measure.label(spots_mask, background=0)
        
        # Initialize cleaned mask
        newspots_mask = np.zeros(spots_mask.shape, dtype=np.uint16)
        
        # Get unique detected spot IDs (excluding background)
        ids = list(np.unique(spots_mask))
        if 0 in ids:
            ids.remove(0)
        
        # Filter spots based on area size
        for spotid in ids:
            spotmask = (spots_mask == spotid).astype(np.uint8)
            spot_area = np.count_nonzero(spotmask)
            
            if spot_area > 5:  # Ignore very small spots
                if spot_area < 20:
                    spotmask = cv2.dilate(spotmask, np.ones((5,5), np.uint8), iterations=1)  # Dilate small spots
                newspots_mask[spots_mask == spotid] = spotid
        
        locus_mask[z] = newspots_mask
    
    # Save the mask
    locus_mask_path = os.path.join(output_path, 'locus_masks', f'{fname}_{locus_id}_mask.tif')
    os.makedirs(os.path.join(output_path, 'locus_masks'), exist_ok=True)
    tifffile.imwrite(locus_mask_path, locus_mask, imagej=True)
    
    return locus_mask


def save_complete_image(cells_im, locus1_norm, locus2_norm, cells_mask, locus1_mask, locus2_mask, output_path, fname):
    """
    Combines the segmented images into a single RGB visualization.

    Parameters:
        cells_im (numpy.ndarray): 3D image of cells.
        locus1_norm (numpy.ndarray): 3D normalized image of first locus.
        locus2_norm (numpy.ndarray): 3D normalized image of second locus.
        cells_mask (numpy.ndarray): 3D mask of cells.
        locus1_mask (numpy.ndarray): 3D mask of first locus.
        locus2_mask (numpy.ndarray): 3D mask of second locus.
        output_path (str): Path where the image will be saved.
        fname (str): Name of the file being processed.
    """
    # Create an empty RGB stack for visualization
    stack_all = np.zeros((*cells_im.shape, 3), dtype=np.uint8)
    stack_all[..., 0] = locus1_norm  # Red channel
    stack_all[..., 1] = locus2_norm  # Green channel
    stack_all[..., 2] = cells_im     # Blue channel

    # Overlay segmentation outlines onto the RGB stack
    for z in range(cells_im.shape[0]):
        for mask, color in zip([locus1_mask, locus2_mask, cells_mask], [(255,0,0), (0,255,0), (255,255,0)]):
            outlines = utils.masks_to_outlines(mask[z])
            outX, outY = np.nonzero(outlines)
            stack_all[z, outX, outY, :] = color
    
    # Save the combined image
    tifffile.imwrite(os.path.join(output_path, f'{fname}_all.tif'), stack_all, imagej=True)


def locus_analysis(cells_mask, locus1_mask, locus2_mask, neurons_ch_flag, neurons_ch_im, cell_id, output_image, mm_to_px, output_path, fname):
    """
    Analysis of the locus in the image-

    Parameters:
        cells_mask (numpy.ndarray): 3D cells mask cleaned.
        locus1_mask (numpy.ndarray): 3D locus1 mask.
        locus2_mask (numpy.ndarray): 3D locus2 mask.
        neurons_ch_im (numpy.ndarray): 3D mask of the 4th channel if exists, otherwise None.
        cell_id (int): id of the interested cell to clean
        output_image (numpy.ndarray): final 3D image with the valid cells.
        mm_to_px (float): micron to pixel conversion value.
        output_path (str): Path where results will be saved.
        fname (str): Name of the file being processed.
    
    Returns:
        output_image (numpy.ndarray): 3D image with the valid cells.
    """
    # Select the mask for the cell_id in analysis and its z section
    cell_mask = np.zeros(cells_mask.shape, dtype=np.uint8)
    cell_mask[cells_mask == cell_id] = 255
    cell_zlayers = [z for z in range(cells_mask.shape[0]) if np.sum(cell_mask[z, cell_mask[z] == 255]) > 0]

    # If there is content in the mask
    if np.sum(cell_mask) > 0:
        masked_l1 = cv2.bitwise_and(locus1_mask, locus1_mask, mask=cell_mask)
        masked_l2 = cv2.bitwise_and(locus2_mask, locus2_mask, mask=cell_mask)
        
        # If there is the 4th channel -> analyse its presence in the cell
        if neurons_ch_flag:
            neurons_ch_ispresent = False
            tot_value = np.average(neurons_ch_im[np.nonzero(neurons_ch_im)])
            masked_im4 = cv2.bitwise_and(neurons_ch_im, neurons_ch_im, mask=cell_mask)
            for z in cell_zlayers:
                cell_value = np.average(masked_im4[z][np.nonzero(masked_im4[z])])
                if cell_value > tot_value:
                    neurons_ch_ispresent = True

        # Select z where there is max 1 spot detected
        l1_zlayers = [z for z in cell_zlayers if len(np.unique(masked_l1[z])) <= 2]
        l2_zlayers = [z for z in cell_zlayers if len(np.unique(masked_l2[z])) <= 2]

        # Verify that at least 1 spot exists in the cell
        l1_exists = [1 for z in cell_zlayers if np.sum(masked_l1[z]) > 0]
        l2_exists = [1 for z in cell_zlayers if np.sum(masked_l2[z]) > 0]

        if cell_zlayers == l1_zlayers and cell_zlayers == l2_zlayers and np.sum(l1_exists) > 0 and np.sum(l2_exists) > 0:
            # Draw and study it
            positions_l1 = [] # list of z,y,x coords of the points
            positions_l2 = []
            
            min_l1_dist = 100000

            min_l2_dist = 100000
            for z in cell_zlayers:
                cell_mask_contours, _ = cv2.findContours(cell_mask[z], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                outlines = utils.masks_to_outlines(cell_mask[z])
                outX, outY = np.nonzero(outlines)
                output_image[z, outX, outY, :] = np.array([0, 153, 255])
                if neurons_ch_flag and neurons_ch_ispresent:
                    output_image[z, outX-2, outY-2, :] = np.array([255,255,0])
                cv2.putText(output_image[z,:,:,:], '{}'.format(int(cell_id)), (int(np.average(np.nonzero(cell_mask[z])[1])), int(np.average(np.nonzero(cell_mask[z])[0]))), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 153, 255), 1)
                
                if np.sum(masked_l1[z]) > 0:
                    positions_l1.append((z, int(np.average(np.nonzero(masked_l1[z])[1])), int(np.average(np.nonzero(masked_l1[z])[0]))))
                    
                    l1 = (int(np.average(np.nonzero(masked_l1[z])[1])), int(np.average(np.nonzero(masked_l1[z])[0])))
                    outlines = utils.masks_to_outlines(masked_l1[z])
                    outX, outY = np.nonzero(outlines)
                    output_image[z, outX, outY, :] = np.array([255,0,0])
                    closestDist = cv2.pointPolygonTest(cell_mask_contours[0], l1, True)
                    if closestDist > 0 and closestDist < min_l1_dist:
                        min_l1_dist = closestDist
                
                if np.sum(masked_l2[z]) > 0:
                    positions_l2.append((z, int(np.average(np.nonzero(masked_l2[z])[1])), int(np.average(np.nonzero(masked_l2[z])[0]))))
                    
                    l2 = (int(np.average(np.nonzero(masked_l2[z])[1])), int(np.average(np.nonzero(masked_l2[z])[0])))
                    outlines = utils.masks_to_outlines(masked_l2[z])
                    outX, outY = np.nonzero(outlines)
                    output_image[z, outX, outY, :] = np.array([0,255,0])
                    closestDist = cv2.pointPolygonTest(cell_mask_contours[0], l2, True)
                    if closestDist > 0 and closestDist < min_l2_dist:
                        min_l2_dist = closestDist


            if min_l1_dist == 100000:
                min_l1_dist = 0
            if min_l2_dist == 100000:
                min_l2_dist = 0
            

            # Check the internal spots have the same position
            # std of x and y position < 0.5% of the image size (around 5pix)
            if np.std([pos[2] for pos in positions_l1]) < locus1_mask.shape[2]*0.005 and np.std([pos[1] for pos in positions_l1]) < locus1_mask.shape[1]*0.005:
                l1_z = np.median([pos[0] for pos in positions_l1])
                l1_y = np.average([pos[1] for pos in positions_l1])
                l1_x = np.average([pos[2] for pos in positions_l1])

                if np.std([pos[2] for pos in positions_l2]) < locus2_mask.shape[2]*0.005 and np.std([pos[1] for pos in positions_l2]) < locus2_mask.shape[1]*0.005:
                    l2_z = np.median([pos[0] for pos in positions_l2])
                    l2_y = np.average([pos[1] for pos in positions_l2])
                    l2_x = np.average([pos[2] for pos in positions_l2])

                    spots_distance = np.linalg.norm(np.array((l1_z, l1_x, l1_y)) - np.array((l2_z, l2_x, l2_y)))

                    # Calculate max cell diameter
                    diams = []
                    for z in cell_zlayers:
                        # Find contours
                        contours, _ = cv2.findContours(cell_mask[z], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # Get the largest contour (assuming it's the circular object)
                        largest_contour = max(contours, key=cv2.contourArea)
                        # Find the minimum enclosing circle
                        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                        diameter = 2 * radius
                        diams.append(round(diameter, 2))

                    # Save the results in the .csv file
                    if neurons_ch_flag:
                        with open(os.path.join(output_path, fname+'.csv'), "a") as file:
                            writer = csv.writer(file)
                            writer.writerow([cell_id, round(spots_distance/mm_to_px,2), round(min_l1_dist/mm_to_px,2), round(min_l2_dist/mm_to_px,2), round(np.max(diams)/mm_to_px,2), neurons_ch_ispresent])
                    else:
                        with open(os.path.join(output_path, fname+'.csv'), "a") as file:
                            writer = csv.writer(file)
                            writer.writerow([cell_id, round(spots_distance/mm_to_px,2), round(min_l1_dist/mm_to_px,2), round(min_l2_dist/mm_to_px,2), round(np.max(diams)/mm_to_px,2)])
        
    return output_image


def main(output_path, image_name, cells_ch_pos, locus1_ch_pos, locus2_ch_pos, avg_cells_size, neurons_ch_pos, cellpose_model_path):
    
    # Read file name
    fname = os.path.basename(image_name).split('.')[0]
    print()
    print('Processing image: '+os.path.basename(image_name).split('.')[0])

    # Create .csv output file
    if neurons_ch_pos != None:
        with open(os.path.join(output_path, fname+'.csv'), "w") as file:
            writer = csv.writer(file)
            writer.writerow(['cell ID', 'spots distance[µm]', 'min locus1-wall distance[µm]', 'min locus2-wall distance[µm]', 'max cell diameter[µm]', '4th ch'])
    else:
        with open(os.path.join(output_path, fname+'.csv'), "w") as file:
            writer = csv.writer(file)
            writer.writerow(['cell ID', 'spots distance[µm]', 'min locus1-wall distance[µm]', 'min locus2-wall distance[µm]', 'max cell diameter[µm]'])

    # Read image resolution, microns to pixel value
    with tifffile.TiffFile(image_name) as tif:
        values = str(tif.pages[0].tags['XResolution']).split(' = ')[1].replace('(','').replace(')','').strip().split(',')
        mm_to_px = int(values[0])/int(values[1])
        
    # Read the images, one per channel: locus1 (Jenisha-red), locus2 (Jenisha-green), cells
    image = tifffile.imread(image_name)
    cells_im = image[:,cells_ch_pos,:,:]
    locus1_im = image[:,locus1_ch_pos,:,:]
    locus2_im = image[:,locus2_ch_pos,:,:]

    # Normalize the images
    cells_norm = cv2.normalize(cells_im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    locus1_norm = cv2.normalize(locus1_im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    locus2_norm = cv2.normalize(locus2_im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    # Segment the cells (and save their mask)
    cells_mask = mask_cells(output_path, fname, cells_norm, avg_cells_size, cellpose_model_path)

    # Segment the locus
    locus1_mask = mask_spots(locus1_norm, output_path, fname, 'l1')
    locus2_mask = mask_spots(locus2_norm, output_path, fname, 'l2')

    # If there if a 4th channel to analyse, read it
    neurons_ch_im = None
    neurons_ch_flag = False
    if neurons_ch_pos != None:
        neurons_ch_flag = True
        neurons_ch_im = image[:,neurons_ch_pos,:,:]
        neurons_ch_im = cv2.normalize(neurons_ch_im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # Save the final image to visualize all the masks together
    #save_complete_image(cells_im, locus1_norm, locus2_norm, neurons_ch_im, cells_mask, locus1_mask, locus2_mask, output_path, fname)

    # Initialize the final image to visualize only the valid masks
    output_image = np.zeros((cells_im.shape[0], cells_im.shape[1], cells_im.shape[2], 3), dtype=np.uint8)    
    output_image[:, :, :, 0] = locus1_norm
    output_image[:, :, :, 1] = locus2_norm
    output_image[:, :, :, 2] = cells_im
    if neurons_ch_flag:
        output_image[:, :, :, 0] += neurons_ch_im
        output_image[:, :, :, 1] += neurons_ch_im

    # Analyse spots in locus image for each cell detected
    print('# Analysing spots')
    for cell_id in tqdm([i for i in list(np.unique(cells_mask)) if i > 0]):
        # Analyse locus into the cell mask
        output_image = locus_analysis(cells_mask, locus1_mask, locus2_mask, neurons_ch_flag, neurons_ch_im, cell_id, output_image, mm_to_px, output_path, fname)

    # Save the final image to visualize only the valid masks
    tifffile.imwrite(os.path.join(output_path,fname+'_masked.tif'), output_image, imagej=True)  


if __name__ == '__main__':
    ######################################################################################
    ## REQUIRED VARIABLES ##

    # Define input and output paths, eg: '/Users/yourname/Documents/DNA_data/'
    # We expect to find .tif images to analyse in the input path
    input_path = 'data-folder-path'
    output_path = 'result-folder-path'

    # Define order of the default channels (REMEMBER THAT THE COUTING STARTS FROM 0 IN PYTHON!)
    cells_ch_pos = 2
    locus1_ch_pos = 0
    locus2_ch_pos = 1

    # Cells average size, default 45
    avg_cells_size = 45
    
    ######################################################################################
    ## ADDITIONAL VARIABLES ##

    # If exists, define position of the 4th channel, otherwise None
    neurons_ch_pos = None

    # Define cellpose retrained model path, otherwise None
    cellpose_model_path = 'retrained-cellpose-model-path'

    ######################################################################################
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Process all the files in input_path that have .tif extention
    for image_name in [f for f in os.listdir(input_path) if '.tif' in f]:
        main(output_path, os.path.join(input_path, image_name), cells_ch_pos, locus1_ch_pos, locus2_ch_pos, avg_cells_size, neurons_ch_pos, cellpose_model_path)
