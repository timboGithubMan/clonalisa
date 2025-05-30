import math
import os
import heapq
import re
import cv2

add_dll_dir = getattr(os, "add_dll_directory", None)
vipsbin = r"\\hg-fs01\research\Wellslab\ToolsAndScripts\wellWellWells\vips-dev-8.14\bin"
os.environ["PATH"] = os.pathsep.join((vipsbin, os.environ["PATH"]))

# add_dll_dir = getattr(os, "add_dll_directory", None)
# vipsbin = r"C:\vips-dev-8.14\bin"
# os.environ["PATH"] = os.pathsep.join((vipsbin, os.environ["PATH"]))

import glob
import string

from natsort import os_sorted
from collections import OrderedDict
from pyvips import Image
from pyvips import GValue

from skimage.io import imread
from skimage import morphology

from tkinter import Tk
from tkinter import filedialog

def select_directory():
    Tk().withdraw()
    selected_directory = filedialog.askdirectory(initialdir='\\\\hg-fs01\\Research\\Wellslab')
    return selected_directory

def get_subdirs(path):
    for root, dirs, files in os.walk(path):
        if files and not dirs and not "cellProfiler_results" in root:
            yield root, os.path.getctime(root)

def get_most_recent_dirs(paths, num_recent=30):
    most_recent_dirs = []
    for path in paths:
        for dirpath, ctime in get_subdirs(path):
            if len(most_recent_dirs) < num_recent:
                heapq.heappush(most_recent_dirs, (ctime, dirpath))
            elif ctime > most_recent_dirs[0][0]:
                heapq.heapreplace(most_recent_dirs, (ctime, dirpath))
    return [dirpath for ctime, dirpath in sorted(most_recent_dirs, reverse=True)]

def check_key(inputFolder):
    try:
        if "\EVOS" not in inputFolder and "/EVOS" not in inputFolder:
            key_path = "//hg-fs01/research/Wellslab/CYTATION/Users/" + "/".join(
                inputFolder.split("\\")[-2].split("__")[:-1])
            if not os.path.exists(key_path + "\\key.txt"):
                key_path = os.path.dirname(inputFolder)
                if not os.path.exists(key_path + "\\key.txt"):
                    key_path = "//hg-fs01/research/Wellslab/EVOS/" + "/".join(
                        inputFolder.split("\\")[-2].split("__")[:-1])
        else:
            key_path = os.path.dirname(inputFolder)

        with open(key_path + "\\key.txt") as f:
            lines = f.readlines()
    except:
        print("no key for " + inputFolder)
        makeKey = input("do you want to make a key?").lower() in ['true', '1', 't', 'y', 'yes']
        if makeKey:
            key = open(os.path.dirname(inputFolder) + "\\key.txt", 'w+')
            key.close()
            os.startfile(os.path.dirname(inputFolder) + "\\key.txt")
            wait = input("enter anything to continue:")
            print("")


def get_filenames_and_labelinfo(inputFolder):
    allFiles = os_sorted([file for file in glob.glob(inputFolder + "/*.*")
                          if ("Thumbs.db" not in os.path.basename(file)
                              and not (("Wells" in os.path.basename(file)) and ("all" in os.path.basename(file)))
                              and not ".txt" in os.path.basename(file)
                              # and not "Z0_Bright" in os.path.basename(file)
                              # and not "Z1_Bright" in os.path.basename(file)
                              # and not "Z2_Bright" in os.path.basename(file)
                              # and not "Z3_Bright" in os.path.basename(file)
                              # and not "Z1_GFP" in os.path.basename(file)
                              # and not "Z2_GFP" in os.path.basename(file)
                              # and not "Z3_GFP" in os.path.basename(file)
                              # and not "Z4_GFP" in os.path.basename(file)
                              # and not "Z0_RFP" in os.path.basename(file)
                              # and not "Z2_RFP" in os.path.basename(file)
                              # and not "Z3_RFP" in os.path.basename(file)
                              # and not "Z4_RFP" in os.path.basename(file)
                              )])

    channels = ["_GFP", "_RFP", "_TRANS", "_CY5", "_BF", "_DAPI", "_Bright Field"]
    files_with_channels = [f for f in allFiles if any(ch in f for ch in channels)]
    allFiles = files_with_channels if files_with_channels else allFiles

    multipleGroups = False

    rowCounts = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0,
        "E": 0,
        "F": 0,
        "G": 0,
        "H": 0
    }

    if "\EVOS" not in inputFolder and "/EVOS" not in inputFolder:
        imagesPerWell = 0
        numCols = 0

        imageLabels = [("_").join(os.path.basename(file).split("_")[:2]) for file in allFiles]
        imageLabels = list(OrderedDict.fromkeys(imageLabels))
        try:
            sampleWell = imageLabels[0].split("_")[0]
        except:
            sampleWell = imageLabels[0]
        for i in range(len(imageLabels)):
            if imageLabels[i].split("_")[0] != sampleWell:
                imagesPerWell = i
                break
        # Extract well labels and order them uniquely
        imageLabels = list(OrderedDict.fromkeys(imageLabels))
        imageLabels = list(OrderedDict.fromkeys([label.split("_")[0] for label in imageLabels]))
        for label in imageLabels:
            row = label[:1]  # Assuming the row is indicated by the first letter
            col = label[1:]  # Assuming the column is indicated by the rest of the string
            rowCounts[row] += 1
        numCols = max(rowCounts.values())
    else:
        wellsGroups = os.path.basename(inputFolder).split("_")[2].split(",")
        if len(wellsGroups) > 1:
            multipleGroups = True

        numWells = 0
        for group in wellsGroups:
            if "-" in group:
                wellsInfo = group.split("-")
                startRow = string.ascii_uppercase.index(wellsInfo[0][0])
                startCol = wellsInfo[0][1:]
                endRow = string.ascii_uppercase.index(wellsInfo[1][0])
                endCol = wellsInfo[1][1:]
                numRows = endRow - startRow + 1
                numCols = abs(int(endCol) - int(startCol)) + 1
                numWells += numRows * numCols
            else:
                numWells += 1

        allTRANS = os_sorted([file for file in allFiles if "TRANS" in os.path.basename(file)])

        numFiles = len(allFiles)
        if (allTRANS):
            numImages = len(allTRANS)
            filesPerImage = max(int(numFiles / numImages), 1)
            imagesPerWell = max(int(numImages / numWells), 1)
        else:
            filesPerImage = 1
            imagesPerWell = max(int(numFiles / numWells), 1)

        imageIndex = 0
        imageLabels = []
        numCols = 1
        for group in wellsGroups:
            if "-" in group:
                wellsInfo = group.split("-")
                startRow = string.ascii_uppercase.index(wellsInfo[0][0])
                startCol = wellsInfo[0][1:]
                endRow = string.ascii_uppercase.index(wellsInfo[1][0])
                endCol = wellsInfo[1][1:]
                numRows = endRow - startRow + 1
                numCols = abs(int(endCol) - int(startCol)) + 1
                numWells = numRows * numCols
            else:
                wellsInfo = group
                startRow = string.ascii_uppercase.index(wellsInfo[0])
                startCol = wellsInfo[1:]
                endRow = startRow
                endCol = startCol
                numRows = 1
                numCols = 1
                numWells = 1

            if int(startCol) > int(startCol):
                if len(wellsInfo) > 2 and wellsInfo[2] == "snaked":
                    print("you cant have startCol > endCol and use snaked")
                if endRow != startRow:
                    print("you cant have startCol > endCol and use multiple rows")
                allFiles[imageIndex:imageIndex + numCols * (imagesPerWell * filesPerImage)] = reversed(
                    allFiles[imageIndex:imageIndex + numCols * (imagesPerWell * filesPerImage)])
                startCol, endCol = endCol, startCol

            for i in range(numWells * imagesPerWell):
                well = string.ascii_uppercase[startRow + math.floor((math.floor(i / imagesPerWell)) / numCols)] + str(
                    int(i / imagesPerWell % numCols + int(startCol)))
                if i % imagesPerWell == 0:
                    imageLabels.append(well)
                else:
                    imageLabels.append("")
                rowCounts[well[0]] += 1

            if len(wellsInfo) > 2 and wellsInfo[2] == "snaked":
                i = numCols * (imagesPerWell * filesPerImage)
                while i < (numWells * (imagesPerWell * filesPerImage)):
                    allFiles[i + imageIndex:i + imageIndex + numCols * (imagesPerWell * filesPerImage)] = reversed(
                        allFiles[i + imageIndex:i + imageIndex + numCols * (imagesPerWell * filesPerImage)])
                    for col in range(1, numCols + 1):
                        allFiles[
                        i + imageIndex + (col - 1) * (imagesPerWell * filesPerImage):i + imageIndex + col * (
                                    imagesPerWell * filesPerImage)] = reversed(
                            allFiles[
                            i + imageIndex + (col - 1) * (imagesPerWell * filesPerImage):i + imageIndex + col * (
                                        imagesPerWell * filesPerImage)])
                    i += numCols * (imagesPerWell * filesPerImage) * 2

            imageIndex += numWells * (imagesPerWell * filesPerImage)

        numCols = max(rowCounts.values()) / imagesPerWell

    unique_columns = set()
    for label in [label for label in imageLabels if label!='']:
        # Assuming label format is 'RowColumn', like 'A1', 'B10', etc.
        row, col = label[0], label[1:]
        unique_columns.add(col)

    numCols = len(unique_columns)

    try:
        if "\EVOS" not in inputFolder and "/EVOS" not in inputFolder:
            key_path = "//hg-fs01/research/Wellslab/CYTATION/Users/" + "/".join(
                inputFolder.split("\\")[-2].split("__")[:-1])
            if not os.path.exists(key_path + "\\key.txt"):
                key_path = os.path.dirname(os.path.dirname(allFiles[0]))
                if not os.path.exists(key_path + "\\key.txt"):
                    key_path = "//hg-fs01/research/Wellslab/EVOS/" + "/".join(
                        inputFolder.split("\\")[-2].split("__")[:-1])
        else:
            key_path = os.path.dirname(os.path.dirname(allFiles[0]))

        with open(key_path + "\\key.txt") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            if "-" in line.split(":")[0]:
                startKeyRow = string.ascii_uppercase.index(line.split(":")[0].split("-")[0][0])
                endKeyRow = string.ascii_uppercase.index(line.split(":")[0].split("-")[1][0])
                startKeyCol = int(line.split(":")[0].split("-")[0][1:])
                endKeyCol = int(line.split(":")[0].split("-")[1][1:])
                numKeyCols = endKeyCol - startKeyCol + 1
                numKeyRows = endKeyRow - startKeyRow + 1
            else:
                startKeyRow = string.ascii_uppercase.index(line.split(":")[0][0])
                endKeyRow = startKeyRow
                startKeyCol = int(line.split(":")[0][1:])
                endKeyCol = startKeyCol
                numKeyCols = 1
                numKeyRows = 1

            for i in range(numKeyRows):
                for j in range(numKeyCols):
                    for k in range(len(imageLabels)):
                        wellStr = string.ascii_uppercase[startKeyRow + i] + str(j + startKeyCol)
                        if imageLabels[k][:len(wellStr)] == wellStr:
                            if len(wellStr) == 2:
                                try:
                                    if not imageLabels[k][2].isdigit():
                                        imageLabels[k] = imageLabels[k] + "\n" + ":".join(line.split(":")[1:])
                                except:
                                    imageLabels[k] = imageLabels[k] + "\n" + ":".join(line.split(":")[1:])
                            else:
                                imageLabels[k] = imageLabels[k] + "\n" + ":".join(line.split(":")[1:])
    except:
        print("skipping key for " + os.path.dirname(allFiles[0]))

    # if multipleGroups:
    #     paired_list = []
    #     current_group = 0
    #     group_label = None
    #     new_imageLabels = [] 

    #     for i in range(0, len(allFiles), filesPerImage):
    #         if imageLabels[int(i/filesPerImage)] != '':
    #             current_group += 1
    #             group_label = imageLabels[int(i/filesPerImage)]
    #             new_imageLabels.append(group_label)

    #         for j in range(filesPerImage):
    #             paired_list.append((current_group, group_label, allFiles[i + j]))

    #     sorted_pairs = natsorted(paired_list, key=lambda x: (x[1], x[0]))
    #     sorted_groups, sorted_group_labels, sorted_allFiles = zip(*sorted_pairs)

    #     imageLabels = new_imageLabels
    #     allFiles = list(sorted_allFiles)

    return allFiles, imageLabels, imagesPerWell, numCols, rowCounts

def fix_image_lists(master_list, image_list, channel_name):
    if not image_list:
        return []

    fixed_list = []

    for master_image in master_list:
        corresponding_image = next((x for x in image_list if f'{master_image}_{channel_name}' in os.path.basename(x)), None)

        if corresponding_image:
            fixed_list.append(image_list.pop(image_list.index(corresponding_image)))
        else:
            fixed_list.append('black')

    return fixed_list

def file_sort_key(filename):
    # Regex pattern to extract the well row letter(s), well column number, and position number
    well_pattern = re.compile(r"([A-Z]+)(\d+)_pos(\d+)")
    match = well_pattern.search(filename)
    if match:
        row_letters, well_num, pos_num = match.groups()
        return (row_letters, int(well_num), int(pos_num))
    return (filename, 0, 0)  # Fallback for filenames that don't match the pattern

def make_images_and_labels(allFiles, imageLabels):
    allImages = []
    allLabelImages = []
    imageWidth = Image.new_from_file(allFiles[0]).width
    imageHeight = Image.new_from_file(allFiles[0]).height

    allBF = [file for file in allFiles if "Bright Field" in os.path.basename(file)]
    allBF += [file for file in allFiles if "TRANS" in os.path.basename(file)]
    allDAPI = [file for file in allFiles if "DAPI" in os.path.basename(file) and not "masks" in os.path.basename(file)]
    allGFP = [file for file in allFiles if "GFP" in os.path.basename(file)]
    allRFP = [file for file in allFiles if "RFP" in os.path.basename(file)]
    allCY5 = [file for file in allFiles if "CY5" in os.path.basename(file)]

    # allBF = [file for file in allFiles if "Bright Field" in os.path.basename(file) and "Z4" in os.path.basename(file)]
    # allBF += [file for file in allFiles if "TRANS" in os.path.basename(file)]
    # allDAPI = [file for file in allFiles if "DAPI" in os.path.basename(file) and not "masks" in os.path.basename(file)]
    # allGFP = [file for file in allFiles if "GFP" in os.path.basename(file) and "Z0" in os.path.basename(file)]
    # allRFP = [file for file in allFiles if "RFP" in os.path.basename(file) and "Z1" in os.path.basename(file)]
    # allCY5 = [file for file in allFiles if "CY5" in os.path.basename(file)]
    #
    # allFiles = allBF + allGFP + allRFP

    if not ("\EVOS" in allFiles[0] or "/EVOS" in allFiles[0]):
        # Adjusted pattern to include the optional Z[digit] part
        pattern = re.compile(r'(.*?_pos\d+Z?\d*)_(BF|Bright Field|TRANS|DAPI|GFP|RFP|CY5)')
        allMaster = sorted(
            set(
                pattern.search(os.path.basename(file)).group(1)
                for file in allFiles
                if pattern.search(os.path.basename(file)) is not None
            ),
            key=file_sort_key
        )

        allBF = fix_image_lists(allMaster, allBF, 'Bright Field')
        allDAPI = fix_image_lists(allMaster, allDAPI, 'DAPI')
        allGFP = fix_image_lists(allMaster, allGFP, 'GFP')
        allRFP = fix_image_lists(allMaster, allRFP, 'RFP')
        allCY5 = fix_image_lists(allMaster, allCY5, 'CY5')

    allBFImages = []
    allDAPIImages = []
    allGFPImages = []
    allRFPImages = []
    allCY5Images = []

    for (files, images) in [(allBF, allBFImages), (allDAPI, allDAPIImages), (allGFP, allGFPImages),
                            (allRFP, allRFPImages), (allCY5, allCY5Images)]:
        for file in files:
            if file == 'black':
                grayscale_image = Image.black(imageWidth, imageHeight)
            else:
                image = Image.new_from_file(file, access="sequential")

                if image.bands == 3:
                    grayscale_image = (image[0] + image[1] + image[2])

                # if the image has only 1 channel, we assume it is already grayscale
                elif image.bands == 1:
                    if ("\EVOS" in allFiles[0] or "/EVOS" in allFiles[0]):
                        image_copy = Image.new_from_file(file, access="sequential")
                        high = max(image_copy.percent(99.9), 1)
                        # image_copy = Image.new_from_file(file, access="sequential")
                        # low = image_copy.percent(2) * 0.5
                        grayscale_image = image * 255/high
                    else:
                        grayscale_image = image / 256

                # if ("\EVOS" in allFiles[0] or "/EVOS" in allFiles[0]):
                #     if grayscale_image.format == "ushort":
                #         grayscale_image = grayscale_image * 255.0 / 4096
                    # low = grayscale_image.percent(0.1)
                    # high = grayscale_image.percent(99.9)
                    # grayscale_image = ((grayscale_image - low) * (255 / (high - low)))


            images.append(grayscale_image.cast("uchar").copy(interpretation="b-w"))

    # if there are no BF, DAPI, GFP, RFP, or CY5 images found, assume all the files are brightfield
    if (len(allBFImages) + len(allDAPIImages) + len(allGFPImages) + len(allRFPImages) + len(allCY5Images)) < 1:
        for file in allFiles:
            allBFImages.append(
                Image.new_from_file(file, access="sequential")[0].copy(interpretation="b-w").cast("uchar"))

    for i in range(len(imageLabels)):
        if imageLabels[i] != "":
            textImage = Image.text(
                imageLabels[i],
                font='Arial',
                width=imageWidth, height=100 * len(imageLabels[i].split("\n")),
                autofit_dpi=True, rgba=False)[0]
            allLabelImages.append(textImage.copy(interpretation="b-w"))

    allDAPIMasksImages = []
    allDAPIMasks = [file for file in allFiles if "DAPI" in os.path.basename(file) and "masks" in os.path.basename(file)]
    if len(allDAPIMasks) == len(allDAPI):
        allDAPIMasks = [file.replace(".tif", "_masks.png") for file in allDAPI]
        for mask_img in allDAPIMasks:
            masks = imread(mask_img)
            selem = morphology.square(5)  
            dilated_image = morphology.dilation(masks, selem)
            outlines = (dilated_image - masks) > 1
            allDAPIMasksImages.append(Image.new_from_array(outlines).cast("uchar").copy(interpretation="b-w"))

    return imageWidth, imageHeight, allBFImages, allDAPIImages, allGFPImages, allRFPImages, allCY5Images, allLabelImages, allDAPIMasksImages


def sort_well_labels(label):
    if label:
        match = re.match(r'([A-Z]+)(\d+)', label)
        if match:
            row, col = match.groups()
            return (row, int(col))

    return ('Z', 99)  # Placeholder for empty labels

def make_and_save_allWells(imageWidth, imageHeight, allBFImages, allDAPIImages, allGFPImages, allRFPImages,
                           allCY5Images, allLabelImages, allDAPIMasksImages, numCols, imagesPerWell, imageLabels,
                           inputFolder, rowCounts, numImages, outputIndividualImages):
    num_channels = sum(1 for lst in [allBFImages, allDAPIImages, allGFPImages, allRFPImages, allCY5Images, allDAPIMasksImages] if lst)
    shrink_factor = 1
    if numImages > 384:
        shrink_factor = 2

    if "Gen5 Image folder" in inputFolder in inputFolder.lower():
        allWells_path = "\\\\hg-fs01\\research\\Wellslab\\CYTATION\\Users\\" + "\\".join(
            inputFolder.split("\\")[-2].split("__"))
        os.makedirs(allWells_path, exist_ok=True)
    else:
        allWells_path = inputFolder

    wellBFImages = []
    wellDAPIImages = []
    wellGFPImages = []
    wellRFPImages = []
    wellCY5Images = []
    wellDAPIMasksImages = []

    # if (outputIndividualImages):
    #     os.makedirs(allWells_path + "/composite_images", exist_ok=True)
    #     for i, image in enumerate(allImages):
    #         image.jpegsave(allWells_path + "/composite_images/" + imageLabels[i].split("\n")[0] + ".jpg", Q=80)

    if imagesPerWell > 1:
        numImageAcross = math.ceil(math.sqrt(imagesPerWell))
        numImageDown = math.ceil(imagesPerWell / numImageAcross)
        for (allImages, wellImages) in [(allBFImages, wellBFImages), (allDAPIImages, wellDAPIImages),
                                        (allGFPImages, wellGFPImages), (allRFPImages, wellRFPImages),
                                        (allCY5Images, wellCY5Images), (allDAPIMasksImages, wellDAPIMasksImages)]:
            for i in range(0, len(allImages), imagesPerWell):
                wellImage = allImages[i:i + imagesPerWell]
                if "\EVOS" in inputFolder or "/EVOS" in inputFolder:
                    j = numImageAcross
                    while j < len(wellImage):
                        wellImage[j:j + numImageAcross] = reversed(wellImage[j:j + numImageAcross])
                        j += numImageAcross * 2
                wellImages.append(Image.arrayjoin(wellImage, across=numImageAcross))
    else:
        numImageAcross = 1
        numImageDown = 1

        wellBFImages = allBFImages
        wellDAPIImages = allDAPIImages
        wellGFPImages = allGFPImages
        wellRFPImages = allRFPImages
        wellCY5Images = allCY5Images
        wellDAPIMasksImages = allDAPIMasksImages

    imageLabels = [label for label in imageLabels if label]
    
    # Step 1: Sort imageLabels in standard tissue culture plate format
    sorted_indices = sorted(range(len(imageLabels)), key=lambda i: sort_well_labels(imageLabels[i]))
    print(imageLabels)
    print(sorted_indices)
    allLabelImages = [allLabelImages[i] for i in sorted_indices]
    imageLabels = [imageLabels[i] for i in sorted_indices]

    # Step 2: Reorganize all images according to the sorted labels
    wellBFImages = [wellBFImages[i] for i in sorted_indices] if wellBFImages else []
    wellDAPIImages = [wellDAPIImages[i] for i in sorted_indices] if wellDAPIImages else []
    wellGFPImages = [wellGFPImages[i] for i in sorted_indices] if wellGFPImages else []
    wellRFPImages = [wellRFPImages[i] for i in sorted_indices] if wellRFPImages else []
    wellCY5Images = [wellCY5Images[i] for i in sorted_indices] if wellCY5Images else []
    wellDAPIMasksImages = [wellDAPIMasksImages[i] for i in sorted_indices] if wellDAPIMasksImages else []

    numWells = len(allLabelImages)
    wellImageWidth = imageWidth * numImageAcross
    wellImageHeight = imageHeight * numImageDown
    numEmptyWells = 0
    i = numWells

    # Create a mapping of rows to their columns
    row_to_columns = {}
    for label in imageLabels:
        if label:  # Skip empty labels
            row, col = label.split("\n")[0][0], int(label.split("\n")[0][1:])
            row_to_columns.setdefault(row, []).append(col)

    # Sort the columns for each row and collect all unique columns
    all_unique_columns = set()
    for row in row_to_columns:
        row_to_columns[row].sort()
        all_unique_columns.update(row_to_columns[row])

    # Convert to a sorted list
    all_columns = sorted(all_unique_columns)

    # Create row structure only for existing rows and initialize with placeholders
    row_structure = {row: [None] * len(all_columns) for row in row_to_columns.keys()}

    # Fill in the actual columns for each row
    for label in imageLabels:
        if label:
            row, col = label.split("\n")[0][0], int(label.split("\n")[0][1:])
            col_index = all_columns.index(col)  # Find the index of the column in the sorted list
            row_structure[row][col_index] = label

    # Flatten the structure and replace None with placeholders
    flat_structure = [item for sublist in row_structure.values() for item in sublist]
    for i, label in enumerate(flat_structure):
        if label is None:
            # Insert placeholders where there's no image
            for wellImages in [wellBFImages, wellDAPIImages, wellGFPImages, wellRFPImages, wellCY5Images, wellDAPIMasksImages]:
                if wellImages:
                    wellImages.insert(i, Image.black(imageWidth, imageHeight))
            allLabelImages.insert(i, Image.text("X", font='Arial', width=100, height=100,
                                                autofit_dpi=True, rgba=False)[0])

    borderImage = Image.black(wellImageWidth, wellImageHeight) + 255
    borderWidth = int(wellImageWidth / 500)
    borderImage = borderImage.draw_rect(0, borderWidth, borderWidth, wellImageWidth - borderWidth - 1,
                                        wellImageHeight - borderWidth - 1, fill=True).cast("uchar").copy(
        interpretation="b-w")
    
    allLabelImages = [borderImage.insert(label_img, 20, 20) for label_img in allLabelImages]

    channel_colors = []
    allWellsImages = []
    for (wellImages, color) in [(wellBFImages, -1), (wellDAPIImages, 65535), (wellGFPImages, 16711935),
                                (wellRFPImages, -16776961), (wellCY5Images, -16776961), (allLabelImages, -1), (wellDAPIMasksImages, -1)]:
        if wellImages:
            allWellsImages.append(Image.arrayjoin(wellImages, across=numCols, shim=0, halign='low', valign='low',
                                                  hspacing=wellImageWidth, vspacing=wellImageHeight).resize(
                1 / shrink_factor))
            channel_colors.append(color)

    filename = (os.path.basename(allWells_path)[-200:] + "_all" + str(numWells) + "Wells" + ".tif")

    # 1 channels for border/label, so 2 "channels" for a single channel image
    if len(channel_colors) > 2:
        channel_info = ""
        for i, color in enumerate(channel_colors):
            channel_info += f"""<Channel ID="Channel:0:{i}" SamplesPerPixel="1" Color="{color}" />"""

        ome = Image.arrayjoin(allWellsImages, across=1)
        # you must make a private copy before modifying image metadata
        ome = ome.copy()
        page_height = allWellsImages[0].height
        ome.set_type(GValue.gint_type, 'page-height', page_height)
        ome.set_type(GValue.gstr_type, "image-description",
                     f"""<?xml version="1.0" encoding="UTF-8"?>
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
            <Image ID="Image:0">
                <!-- Minimum required fields about image dimensions -->
                <Pixels DimensionOrder="XYCZT"
                        ID="Pixels:0"
                        SizeC="{len(allWellsImages)}"
                        SizeT="1"
                        SizeX="{allWellsImages[0].width}"
                        SizeY="{page_height}"
                        SizeZ="1"
                        Type="uint8">
                        {channel_info}
                </Pixels>
            </Image>
        </OME>""")
        # note: Q factor doesnt change anything for jp2k?
        ome.tiffsave(allWells_path + "/" + filename, pyramid=True, compression="lzw", tile=True, tile_width=512, tile_height=512)
        if openImagesOnCompletion:
            os.startfile(allWells_path + "/" + filename)
    else:
        (allWellsImages[0] + allWellsImages[1]).cast("uchar").jpegsave(
            allWells_path + "/" + filename[:-4] + ".jpg", Q=80)
        if openImagesOnCompletion:
            os.startfile(allWells_path + "/" + filename[:-4] + ".jpg")


# paths = [r"\\hg-fs01\research\Wellslab\EVOS", r"C:\Users\Imaging Controller\Desktop\Gen5 Image folder"]
# lowest_dirs = get_most_recent_dirs(paths)
# for i, folder in enumerate(reversed(lowest_dirs[:30])):
#     print(str(30 - i) + " - " + folder[33:])

while True:
    selectedDirsInput = select_directory().replace('/', '\\')
    # openImagesOnCompletion = False
    # openImagesOnCompletion = input('Open images on completion?:').lower() in ['true', '1', 't', 'y', 'yes', '']
    openImagesOnCompletion = r'/EVOS' in selectedDirsInput or r'\EVOS' in selectedDirsInput
    # outputIndividualImages = input('Output individual compressed images?:').lower() in ['true', '1', 't', 'y', 'yes']
    outputIndividualImages = True

    selectedDirs = []

    if "\\" in selectedDirsInput:
        for root, dirs, files in os.walk(selectedDirsInput):
            if files and not dirs:
                selectedDirs.append(root)

    for folder in selectedDirs:
        check_key(folder)

    for folder in selectedDirs:
        # try:
        allFiles, imageLabels, imagesPerWell, numCols, rowCounts = get_filenames_and_labelinfo(folder)
        imageWidth, imageHeight, allBFImages, allDAPIImages, allGFPImages, allRFPImages, allCY5Images, allLabelImages, allDAPIMasksImages = make_images_and_labels(
            allFiles, imageLabels)
        make_and_save_allWells(imageWidth, imageHeight, allBFImages, allDAPIImages, allGFPImages, allRFPImages,
                               allCY5Images, allLabelImages, allDAPIMasksImages, numCols, imagesPerWell, imageLabels,
                               folder, rowCounts, len(allFiles), outputIndividualImages)
        print("finished " + folder)
