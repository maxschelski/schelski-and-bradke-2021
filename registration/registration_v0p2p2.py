# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:22:35 2019

@author: schelskim
"""

import os
import tifffile
from skimage import io, feature
from scipy.signal import fftconvolve
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL.TiffTags import TAGS as tiff_tags_dict
from collections import OrderedDict
from skimage.filters import median
from skimage.morphology import disk
import time
import tkinter
from tkinter import filedialog
from PIL import Image, ImageSequence
import copy
import re
from scipy.ndimage import median_filter
import pandas as pd
import itertools

#sccript registered two channels based on reference channel (folder name of channel)

#how to implement vectorization?
#do alignment for all images wth the same reference image together.
#create one more dimension in the vector for that
#then create one more dimension each shifted image
#in multi dimensional alignment value matrix then exclude ids where best alignment is not at the border
#continue with remaining array until no id is left

#how to solve the problem of huge shifts in a short time interval?
#if it is over a few frames, reduce number of times that the same reference image is used
#if it is from one timeframe to the next:
#create mask of reference image and new timeframe
#check overlap of the two masks relative to the smaller of the two masks
#if it is below X % (50?) move masks somehow to get above this threshold
#maybe move it based on centroid of image?
#for that first make size of both masks similar in each dimension:
#recursively: check centroid, cut in each dimension a few percent
#(not of are but of coordinates), restart
#something like this might be necessary, since
# after a big step of movement, some part of the cell might be cut off, shifting the centroid
#is there a more clever algorithm to still get the middle point? hmmm...


#is it possible to create an ML algorithm that detects neurites
#by using the algorithm I coded for automated analysis?
#Since it might average out random mistakes. Hm...




#collectionFolder = "C:\\01DATA\\TUBB\\02PROCESSED\\"
#collectionFolder = "C:\\01DATA\\TUBB\\automated-analysis\\MT-fluctuations\\"
collectionFolder = "C:\\Users\\Maxsc\\Documents\\01DATA\\TUBB\\MT-RF\\"
collectionFolder = "E:\\00DATA\\TUBB\\TO_SORT\\"
collectionFolder = "E:\\TUBB\\MT-RF_longterm\\CAMS3-longterm\\TUBB\\"
collectionFolder = "E:\\TUBB\\MT-RF_manipulate_longterm\\FRB-Lifact-bicDN-FKBP-longterm\\"
collectionFolder = "E:\\TUBB\\01toBeProcessed\\210521\\MTRF_manipulation_longterm\\sorted\\recruit-FRB-KIFC1-593-Dyn1-C2-longterm\\"
collectionFolder = "E:\\TUBB\\02PROCESSED\\"
collectionFolder = "E:\\TUBB\\MT-RF_longterm\\CAMS3-longterm\\cytosol_40x\\"
collectionFolder = "F:\\Max\\to_register\\"
#Choose the channel with an as much as possible non-changing intracellular localization
#signals that accumulate at a certain place or are excluded from a from a place
#over time are not going to work very well
referenceChannel = "c0000"
date = "2101"
imagesInSortedFolder = True
chooseFolderManually = False
force_multipage_save = False
start_reg_from_last_img = False
only_delete_out_of_focus_at_end = False
overwrite_registered_images = True
only_register_multipage_tiffs = False
nb_empty_frames_instead_of_out_of_focus_frames = 0

processes = 1
process = 1
background_val = 200


def getNormalizedImage(image, image_mean):
    image_normalized = image - image_mean
    image_normalized = np.array(image_normalized.astype(int))
    return image_normalized


def get_image_properties(file_path):
    """
    Extracts order of dimensions of imageJ image and image width and height
    """
    with Image.open(file_path) as img:
        dimensions = ["image", "frames", "slices", "channels"]

        #get key that is used for imagedescription in ".tag" dict
        tiff_tags_inv_dict = {v:k for k,v in tiff_tags_dict.items()}
        tiff_tag = tiff_tags_inv_dict["ImageDescription"]
        #use create ordered dict of imagedescription
        #order in dict determines which dimension in the array is used
        #counting starts from right
        meta_data_str = img.tag[tiff_tag][0]
        meta_data = OrderedDict()
        data_values = meta_data_str.split("\n")
        data_dict = OrderedDict()
        for value in data_values:
            value_split = value.split("=")
            if len(value_split) == 2:
                if value_split[0] in dimensions:
                    data_dict[value_split[0]] = value_split[1]
            if len(value_split) > 1:
                meta_data[value_split[0]] = value_split[1]
        img_width = np.array(img).shape[1]
        img_height = np.array(img).shape[0]
        data_dict = OrderedDict(reversed(list(data_dict.items())))
    return data_dict, img_width, img_height, meta_data


def move_xy_axes_in_img_to_last_dimensions(img, img_width, img_height):
    """
    Move x and y axes of all images in stack (img) to last position in dimension list
    :param img: multi-dimensional numpy array
    :param img_width: width (length of x axis) of one image in img (stack)
    :param img_height: height (length of y axis) of one image in img (stack)
    """
    img_axes = img.shape
    #if there are more than 2 (x,y) axes
    if len(img_axes) > 2:
        #check which axes are x and y and put these axes last
        xy_axes = []
        for ax_nb, axis in enumerate(img_axes):
            if (axis == img_width) | (axis == img_height):
                xy_axes.append(ax_nb)
        for xy_nb, xy_axis in enumerate( reversed(xy_axes) ):
            img = np.moveaxis(img, xy_axis, - xy_nb - 1 )
    return img


def getImagesFromFolder(folder):
    """
    Get all images in one folder.
    Each image should only contain images of the same cell and channel
    Save each plane from a multipage image (with different timepoints) in
    array
    """
    input_image_array = []
    inputImageNameArray = []
    imageShape = []
    imageShapeSet = False
    image_names = os.listdir(folder)
    multi_page = False
    for inputImageName in image_names:
        if inputImageName.find(".tif") != -1:
            #go through each file and if it is a multi page
            #than go through each plane and add it separately
            #NOT IMPLEMENTED: MULTIPAGE!!!
            inputImageNameArray.append(inputImageName)
            input_image_path = os.path.join(folder, inputImageName)
            input_image_array.append(io.imread(input_image_path))
            single_plane = np.array(input_image_array[0])
            imageShape = single_plane.shape

    return np.array(input_image_array), inputImageNameArray, imageShape, multi_page #inputImageArray replaced by input_image


def extract_imgs_from_multipage(path, image_name):
    """
    Extract all images from multipage ImageJ image.
    Allow image to contain several channels as well.
    For each channel add array of all images, all names, imageshape and the channel number.
    """
    image_path = os.path.join(path,image_name)

    input_image = io.imread(image_path)
    #name of channel property in data_dict, according to imageJ
    channel_prop = "channels"

    data_dict, img_width, img_height, meta_data = get_image_properties(image_path)

    input_image = move_xy_axes_in_img_to_last_dimensions(input_image, img_width, img_height)

    #get dimension number of channel
    #convert ordered dict keys object to list to access index attribute
    channel_dim = list(data_dict.keys()).index(channel_prop)

    #create slice object to reference to entire array of only one channel
    slices = [slice(None)] * (len(data_dict) + 2)


    image_arrays = []
    image_name_arrays = []
    image_shapes= []
    channels= []

    #iterate through each channel
    for channel in range(int(data_dict[channel_prop])):

        #create slice object to get all images of one channel
        slices_channel = copy.copy(slices)
        slices_channel[channel_dim] = channel
        image_array = input_image[slices_channel]
        image_name_channel = image_name.replace(".tif","c"+str(channel)+".tif")

        image_arrays.append(image_array)
        image_name_arrays.append([image_name_channel])
        image_shapes.append([img_height, img_width])
        channels.append("c{0:0=4d}".format(channel))

    return image_arrays, image_name_arrays, image_shapes, channels, meta_data


def delete_out_of_focus_images(imageArrays, channel, cell, background_val, deleted_frames_cols, output_folder, nb_empty_frames_instead_of_out_of_focus_frames):
    #remove frames out of focus in both channels
    #use referenceChannel since it has a stronger, more consistent signal
    image = imageArrays[channel]
    #create mask of all px in 4D array above background
    image_mask = image > background_val
    #create zero matrix with 1s only at points above background
    image_thresh = np.zeros_like(image)
    image_thresh[image_mask] = 1
    #calculate the number of px above background for each timeframe
    nb_px_above_background = np.sum(image_thresh, axis=(1,2))
    #set zero values as 1 to prevent division by zero
    nb_px_above_background[nb_px_above_background == 0] = 1
    #calculate the relative change in the number of px above background
    change_px_above_background = nb_px_above_background[1:] / nb_px_above_background[:-1]
    #get positions at which focus was probably lost and where it was gained again
    loss_of_focus = np.where(change_px_above_background < 0.2)[0]
    gain_of_focus = np.where(change_px_above_background > 5)[0]
    #get all ranges where focus is lost
    lost_focus_ranges = []
    position = 0
    while True:
        starts_of_ranges = loss_of_focus[loss_of_focus > position]
        if len(starts_of_ranges) > 0:
            #set first value in startof range (smallest)
            #as start of range for this iteration
            start_of_range = starts_of_ranges[0]
            ends_of_ranges = gain_of_focus[gain_of_focus > start_of_range]
            if len(ends_of_ranges) > 0:
                for end_of_range in ends_of_ranges:
                    #check if there is another loss of focus before the next gain of focus
                    all_losses_of_focus_between = loss_of_focus[(loss_of_focus < ends_of_ranges[0]) & (loss_of_focus > start_of_range)]
                    if len(loss_of_focus) == 0:
                        break
                    else:
                        #if there is, multiply the changes in nb of px of
                        #initial loss of focus events and all in between loss of focus events
                        complete_loss_of_focus = change_px_above_background[start_of_range]
                        for loss_of_focus_between in all_losses_of_focus_between:
                            complete_loss_of_focus *= change_px_above_background[loss_of_focus_between]
                        #check if the next gain of focus event multiplied by total loss of focus
                        #and multiplied by 2 (make up for errors) is more than 1
                        complete_gain_of_focus = change_px_above_background[end_of_range]
                        if (2 * complete_gain_of_focus * complete_loss_of_focus) >= 1:
                            break
                        #if not, multiply total loss of focus by gain of focus
                        else:
                            complete_loss_of_focus *= end_of_range
                        #then move to next gain of focus (ends_of_ranges) event and repeat

                position = end_of_range
                lost_focus_ranges.append([start_of_range+1, end_of_range+1])
            else:
                #if there is no gain of focus after its lost,
                #then focus is lost till the end of the timeseries
                lost_focus_ranges.append([start_of_range+1, len(nb_px_above_background)])
                break
        else:
            break
    #reverse lost focus range to start removing images from the back

    lost_focus_ranges.reverse()
    all_deleted_indices = []
    #go through each range of lost focus
    for lost_focus_range in lost_focus_ranges:
        #create a list of all indices with lost focus for current range
        lost_focus_all_ind = list(range(lost_focus_range[0],lost_focus_range[1]))
        #save all indices of frames that will be deleted
        all_deleted_indices.append(lost_focus_all_ind)

        #extract first X nb of frames in frames to delete
        #X = nb_empty_frames_between_instead_of_out_of_focus_frames
        #max size of empty images has to be the number of out of focus images in current range
        nb_empty_frames_instead_of_out_of_focus_frames = min(nb_empty_frames_instead_of_out_of_focus_frames, len(lost_focus_all_ind))
        slices_to_replace_with_empty_frames = slice(nb_empty_frames_instead_of_out_of_focus_frames)
        indices_to_replace_with_empty_frames = lost_focus_all_ind[slices_to_replace_with_empty_frames]
        lost_focus_all_ind = np.delete(lost_focus_all_ind, slices_to_replace_with_empty_frames)

        #delete all indices for both channels
        for nb, image_array in enumerate(imageArrays):
            imageArrays[nb] = np.delete(imageArrays[nb], lost_focus_all_ind, axis=0)

            #then replace indices defined above with empty frames
            #(nb_empty_frames_between_instead_of_out_of_focus_frames)
            for index_to_replace_with_empty_frames in indices_to_replace_with_empty_frames:
                imageArrays[nb][index_to_replace_with_empty_frames,:,:] = np.zeros_like(imageArrays[nb][0,:,:])

        #if only out of focus images at end should be delete
        #stop after one iteration of for loop
        #thereby each video will only go until the last in focus image
        if only_delete_out_of_focus_at_end:
            break
    #flatten list with list of all indices that were deleted
    all_deleted_indices = list(itertools.chain(*all_deleted_indices))

    #save all frames that were deleted in dataframe
    deleted_frames = pd.DataFrame(columns=deleted_frames_cols)
    deleted_frames["frames"] = all_deleted_indices
    deleted_frames["date"] = date
    deleted_frames["cell"] = cell
    csv_file_path = os.path.join(output_folder, cell.replace(".tif","") + "_df.csv")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    deleted_frames.to_csv(csv_file_path)
    return imageArrays



def calculate_correlation_value(inputImage, xShift, yShift, input_image_std, input_image_mean, referenceImage, stdReference):
    shiftedImage = np.roll(inputImage,(xShift,yShift),axis=(1,0))
    shiftedImage = getNormalizedImage(shiftedImage, input_image_mean)
    imageProduct = np.multiply(shiftedImage, referenceImage)
    imageProduct = imageProduct / (stdReference * input_image_std)
    imageProductSum = np.sum(imageProduct)
    correlationValue = imageProductSum / (inputImage.shape[0] * inputImage.shape[1])
    return correlationValue

def getTranslations(inputImageArray, step_size_shift):

    allShifts = []
    nbUntilNewRef = 70
    is_first_nonzero_image = True
    lastXShift = 0
    lastYShift = 0
    """
    #WORK IN PROGRESS TO VECTORIZE THE REGISTRATION
    start = time.time()
    # ref_norm = inputImageArray[0] - np.mean(inputImageArray[0])
    # std_ref = np.std(ref_norm)

    refs_norm = norm[:-1, :, :]
    stds = np.std(norm, axis=(1,2))
    stds_ref = stds[:-1]
    norm = norm[1:, :, :]
    stds = stds[1:]
    print(time.time() - start)

    start = time.time()
    for x_shift in range(3):
        for y_shift in range(3):
            rolled = np.roll(norm, (x_shift, y_shift), axis=(1,2))
            products = np.multiply(rolled, refs_norm)
            products = products / (stds * stds_ref)[:, None, None]
            sums = np.sum(products, axis=(1,2))
            correlation_values = sums / (inputImageArray[0].shape[0] * inputImageArray[0].shape[1])
    print(time.time() - start)
    """



    if start_reg_from_last_img:
        inputImageArray = np.flip(inputImageArray, axis=0)
    else:
        allShifts.append([0,0])


    counter = 0
    for a, inputImage in enumerate(inputImageArray):
        print(a)
        if is_first_nonzero_image:
            if len(np.unique(inputImage)) > 1:
                is_first_nonzero_image = False
                referenceImage = getNormalizedImage(inputImage, np.mean(inputImage))
                stdReference = np.std(referenceImage)
                maxNumberOfCorrelationRounds = int(np.round(referenceImage.shape[0]/2 / step_size_shift,0))
            elif a > 0:
                allShifts.append([0,0])
        else:
            if len(np.unique(inputImage)) == 1:
                allShifts.append([0,0])
            else:
                input_image_mean = np.mean(inputImage)
                std_input_image = np.std(inputImage)
                #shifting too far could cause problems due to rolling of image (stuff from one side would appear on other side)
                #also big shifts from one to the next frame would cause a linear slow down of the algorithm
                #check max number of trials (shifts of lastXShift) to perform

                #for each iteration calculate a new correlation value matrix
                correlationValueArray = np.zeros((step_size_shift*2 + 1, step_size_shift*2 + 1))
                for xShift in range(lastXShift - step_size_shift, lastXShift + step_size_shift + 1):
                    for yShift in range(lastYShift - step_size_shift, lastYShift + step_size_shift + 1):
                        correlationValue = calculate_correlation_value(inputImage, xShift, yShift, std_input_image, input_image_mean, referenceImage, stdReference)
                        correlationValueArray[xShift + step_size_shift - lastXShift, yShift + step_size_shift - lastYShift] = correlationValue

                #get the highest (best) correlation value
                best_correlation_value = np.max(correlationValueArray)
                #get the position of the best correlation value in matrix
                bestCorrelation = np.where(correlationValueArray == best_correlation_value)

                #calculate corresponding shifts by starting from lastshifts
                xShift = bestCorrelation[0][0] - step_size_shift + lastXShift
                yShift = bestCorrelation[1][0] - step_size_shift + lastYShift

                #check whether shift should be refined in each direction
                #if best value was not +1 or -1 px, leave it as is
                #if the best value was left in the matrix (lower shift)
                #then the shift should be further reduced
                #if it was right in the array (higher shift)
                #then the shift should be further increased
                refineXShift = True
                refineYShift = True
                if (bestCorrelation[0][0] == 0):
                    #reduce xShift
                    x_shift_change = -1
                elif (bestCorrelation[0][0] == (2 * step_size_shift)):
                    #increase xShift
                    x_shift_change = 1
                else:
                    refineXShift = False

                if (bestCorrelation[1][0] == 0):
                    y_shift_change = -1
                elif (bestCorrelation[1][0] == (2 * step_size_shift)):
                    y_shift_change = 1
                else:
                    refineYShift = False

                # start = time.time()
                while refineXShift | refineYShift:
                    #save all xShifts & yShifts to be tested
                    xShifts = []
                    xShifts.append(xShift)
                    yShifts = []
                    yShifts.append(yShift)
                    #if shift should be refined
                    #change shift further in previously defined direction
                    if refineXShift:
                        xShifts.append(xShift + x_shift_change)
                    if refineYShift:
                        yShifts.append(yShift + y_shift_change)

                    xShift_tmp = xShift
                    yShift_tmp = yShift
                    #test all combinations of xShifts and yShifts
                    for xShift_test in xShifts:
                        for yShift_test in yShifts:
                            #don't calculate correlation that was calculated already again
                            if (xShift_test != xShift) | (yShift_test != yShift):
                                correlation_value_test = calculate_correlation_value(inputImage, xShift_test, yShift_test, std_input_image, input_image_mean, referenceImage, stdReference)
                                #check if the new correlation_value is larger than the best so far
                                if correlation_value_test > best_correlation_value:
                                    #update best correlation value
                                    best_correlation_value = correlation_value_test
                                    #save shifts for best correlation value in tmp vars
                                    xShift_tmp = xShift_test
                                    yShift_tmp = yShift_test

                    #if x_shift / y_shift were not changed, then don't refine it further
                    #otherwise update the shifts
                    if xShift_tmp == xShift:
                        refineXShift = False
                    else:
                        xShift = xShift_tmp

                    if yShift_tmp == yShift:
                        refineYShift = False
                    else:
                        yShift = yShift_tmp

                allShifts.append([xShift,yShift])
                #update last shift variables
                #will be used as starting point for shifting the next frame
                lastXShift = xShift
                lastYShift = yShift

                # print("while",time.time() - start)
                if len(np.unique(inputImage)) > 1:
                    #do vector addition on last shifts
                    #to check the total shift that was performed over the last images
                    # last_shifts = allShifts[-4:]
                    # summed_last_shifts = np.sum(last_shifts)
                    # if np.sum(summed_last_shifts):
                    define_new_ref_image = False
                    if (counter >= nbUntilNewRef):
                        define_new_ref_image = True

                    if define_new_ref_image:
                        counter = 1
                        referenceImage = np.roll(inputImage,(xShift,yShift),axis=(1,0))
                        referenceImage = getNormalizedImage(referenceImage, input_image_mean)
                        stdReference = std_input_image

                    else:
                        counter += 1

    if start_reg_from_last_img:
        allShifts.reverse()
        allShifts.append([0,0])

    return allShifts

def translateImages(allTranslations,imageArray,imageNameArray, outputFolder, channel, save_as_multipage=False):

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    registered_images_array = []
    for a, image in enumerate(imageArray):
        if a < len(allTranslations):
            xShift = allTranslations[a][0]
            yShift = allTranslations[a][1]
            translatedImage = np.roll(image,(xShift,yShift),axis=(1,0))
            if xShift > 0:
                translatedImage[:,:xShift] = 0
            if xShift < 0:
                translatedImage[:,xShift:] = 0
            if yShift > 0:
                translatedImage[:yShift,:] = 0
            if yShift < 0:
                translatedImage[yShift:,:] = 0

            #only save single images if they should not be saved as multipage images
            if not save_as_multipage:
                io.imsave(os.path.join(outputFolder,imageNameArray[a]),translatedImage)
            else:
                registered_images_array.append(translatedImage)#Image.fromarray(
    #if tiff should be saved as multi page, do so instead of saveing single images
    if save_as_multipage:
        registered_images_array = np.array(registered_images_array)
        # registered_images_array[0].save(outputFolder+imageNameArray[0], compression="tiff_deflate",
        #                                 save_all=True, append_images=registered_images_array[1:])
    return registered_images_array


def translateCellsInFolder(expIterationFolder, date, step_size_shift,current_nb, force_multipage_save = False) :
    """
    Translate all cells within a folder starting with xy shift values up until step_size_shift.
    Allow cells within folder to be either saved as tiff files in channel folder in cell folder
    or cells saved as a single multi-channel tiff
    """
    for cell in os.listdir(expIterationFolder):
        imageArrays = []
        imageNameArrays = []
        imageShapes = []
        channels = []
        cellFolder = os.path.join(expIterationFolder,cell)
        cell_folder_reg = os.path.join(expIterationFolder,cell+"_registered")
        #check if registration was done either with or without multipage save
        registration_done_no_multipage = os.path.exists(cell_folder_reg)
        cell_file_name_multipage = cell.replace(".tif","_registered.tif")
        registration_done_multipage = os.path.exists(os.path.join(expIterationFolder, cell_file_name_multipage))
        registration_done = registration_done_no_multipage | registration_done_multipage
        if ((cell.replace("_registered","") == cell) & ~(registration_done) ) | overwrite_registered_images:
            output_folder = os.path.join(expIterationFolder,cell + "_registered")
            if not only_register_multipage_tiffs:
                if os.path.isdir(cellFolder) & (cell.find("cell0") != -1):
                    current_nb += 1
                    print("cell folder:",cell)
                    if current_nb == process:
                        for channel in os.listdir(cellFolder):
                                channelFolder = os.path.join(cellFolder,channel)
                                if os.path.isdir(channelFolder):
                                    imageArray, imageNameArray, imageShape, multi_page = getImagesFromFolder(channelFolder)
                                    imageArrays.append(imageArray)
                                    imageNameArrays.append(imageNameArray)
                                    imageShapes.append(imageShape)
                                    channels.append(channel)


            if (cell.find(".tif") != -1) & (len(imageArrays) == 0):
                output_folder = expIterationFolder
                current_nb += 1
                print("tiffile:",cell)
                if (current_nb == process):
                    #if item in expIterationFolder is not a folder
                    #and is tiff or tif file, then process as multipage tiff
                    imageArrays, imageNameArrays, imageShapes, channels, meta_data = extract_imgs_from_multipage(expIterationFolder, cell)
                    multi_page = True

        #independent of whether cell data is saved in folder structure
        #or in multi-channel tiff, get translations from reference channel
        #and then translate all images accordingly
        if len(imageArrays) > 0:
            if len(imageArrays[0].shape) > 2:
                for nb, channel in enumerate(channels):
                    #remove zero images
                    #by selecting all timeframes comprised of a non zero image
                    #imageArrays[nb] = imageArrays[nb][np.all(imageArrays[nb] > 0, axis=(1,2)),:,:]
                    if channel == referenceChannel:
                        imageArrays = delete_out_of_focus_images(imageArrays, nb, cell, background_val, deleted_frames_cols, output_folder, nb_empty_frames_instead_of_out_of_focus_frames)

                for nb, channel in enumerate(channels):
                    if channel == referenceChannel:
                        allTranslations = getTranslations(imageArrays[nb], step_size_shift)


                registered_images_array_all_channels = []
                if multi_page:
                    output_folder_channel = expIterationFolder

                for nb, channel in enumerate(channels):
                        if force_multipage_save:
                            multi_page = True

                        if not multi_page:
                            output_folder_channel = os.path.join(output_folder,channel)
                            if not os.path.exists(output_folder_channel):
                                os.makedirs(output_folder_channel)
                                
                        registered_images_array = translateImages(allTranslations, imageArrays[nb], imageNameArrays[nb], output_folder_channel, channel, save_as_multipage = multi_page)
                        
                        if type(registered_images_array) != type(None):
                            registered_images_array_all_channels.append(registered_images_array)

                #only save as multipage tiff if meta data was extracted previously
                if (len(registered_images_array_all_channels) > 0) & ("meta_data" in locals()):
                    #transform list into numpy array
                    registered_images_array_all_channels = np.array(registered_images_array_all_channels)
                    #axes need to be arranged as TZCYX to be displayed as correct Hyperstack in ImageJ
                    #first move axes for time to first position
                    registered_images_array_all_channels = np.moveaxis(registered_images_array_all_channels, 1,0)
                    #then add empty axes for Z dimension at second position
                    registered_images_array_all_channels = np.expand_dims(registered_images_array_all_channels, 1)
                    #add all ImageJ meta_data from the non registered file as well
                    image_path = os.path.join(output_folder, cell.replace(".tif","_registered.tif"))
                    io.imsave(image_path , registered_images_array_all_channels, imagej=True,plugin="tifffile", metadata = meta_data)
                        
        if (current_nb % processes) == 0:
            current_nb = 0
    return current_nb

#expected shift needs to be at least 1
step_size_shift = 3

current_nb = 0

deleted_frames_cols = ("date", "cell", "frame")

if chooseFolderManually:
    print("choose folder now...")
    root = tkinter.Tk()
    root.withdraw()

    collectionFolder = filedialog.askdirectory(initialdir = collectionFolder)
    collectionFolder = os.path.abspath(collectionFolder)

    if os.path.exists(collectionFolder):
        if os.path.isdir(collectionFolder):
            collectionFolder = collectionFolder

            current_nb = 0
            translateCellsInFolder(collectionFolder, "unknown", step_size_shift, current_nb, force_multipage_save = force_multipage_save)

else:
    for date in os.listdir(collectionFolder):
        print(date)
        dateFolder = os.path.join(collectionFolder, date)
        if(os.path.isdir(dateFolder)):
            for expClass in os.listdir(dateFolder):
                print(expClass)
                if imagesInSortedFolder:
                    expClassFolder = os.path.join(dateFolder,expClass,"sorted")
                else:
                    expClassFolder = os.path.join(dateFolder, expClass)
                if(os.path.isdir(expClassFolder)):
                    for expType in os.listdir(expClassFolder):
                        print(expType)
                        expTypeFolder = os.path.join(expClassFolder, expType)
                        if(os.path.isdir(expTypeFolder)):
                            for expIteration in os.listdir(expTypeFolder):
                                print(expIteration)
                                expIterationFolder = os.path.join(expTypeFolder, expIteration)
                                if os.path.isdir(expIterationFolder):
                                    current_nb = translateCellsInFolder(expIterationFolder, date, step_size_shift, current_nb, force_multipage_save = force_multipage_save)
