import cv2
import numpy as np
import os

#location of Pictures
pictures = "C:\\Users\\gerwi\\Dropbox\\uni\\BA_Erdbeeren\\Datasets\\exp\\crops\\"

#constants of HVS colours to find strawberries
mycolors = [
          [np.array([0,50,55]),np.array([13,255,255]), 4],     #"red": color found from testenv.py using colors.jpg
          [np.array([0,121,67]),np.array([7,255,255]), 4],    #"red1": color found from testenv.py using right000005.jpg
          [np.array([0,189,105]),np.array([20,255,255]), 4],  #"red2": color found from testenv.py using right0004065.jpg
          #[np.array([0,205,69]),np.array([179,255,255]), 4],  #"red3": color found from testenv.py using right0004065.jpg
          [np.array([14,92,119]),np.array([35,200,255]), 2], #"green": color found from testenv.py using pics in StrawberyNotReady folder
          [np.array([0,34,67]),np.array([56,255,255]), 2],   #"green1": colour found from testenv.py using right0001963.jpg
          [np.array([0,5,0]),np.array([61,229,150]), 1],
          [np.array([32,10,42]),np.array([74,220,178]), 2],
          [np.array([15,84,47]),np.array([104,224,223]), 3],
          [np.array([3,17,162]),np.array([25,151,253]), 3],
          [np.array([2,57,69]),np.array([19,211,255]), 3],
          [np.array([0,121,67]),np.array([7,255,255]), 4],
          [np.array([0,198,103]),np.array([198,255,255]), 4],
          [np.array([2,137,103]),np.array([34,255,239]), 5]
          ]

#define constants:
pi = 3.1415
pix = 1/50   #convert from pixel to cm -> 1 cm = 1/p Pixel
dens = 0.858  #Density of a  strawberry in g/cm³

'''preprocess function: 
Processes raw strawberry picture, finds all the contours and picks (??? and combines ???)
the best one by choosing the biggest then turnes it into useable format (black and white)
Input: Foto, list-colorsetting
Output: Image(red and black, (700,700))'''
def getrgbcontourimage(img, colors):
    ''' Function processes Image into a black and white shape of the original,
    uncomment the linecomment with join, to get all the contours find with the parameters in colors'''
    dif = {}  #stores area as key and contour as value
    join = np.zeros((img.shape[0],1), np.uint8)
    for col in range(len(colors)):
########find strawberry in hsv plane
        hsvimage = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        blurhsv = cv2.GaussianBlur(hsvimage, (5, 5), 0)  # optional, does not do that much, initial (5,5)
        mask = cv2.inRange(blurhsv, colors[col][0], colors[col][1])
        edgeimage = cv2.Canny(mask, 50, 100, 20)
        kernel = np.ones((2, 2))  # initial 5,5 smaler kernel works better with smaller pictures
        dialimage = cv2.dilate(edgeimage, kernel, iterations=2)  # initial 2
        first = cv2.erode(dialimage, kernel, iterations=1)  # initial 1
        join = np.hstack((join, first))
        cv2.imshow("finder", join); cv2.waitKey(0)

########find maximum contour of this colour setting:
        contours, hierachy = cv2.findContours(first, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = {}
        try:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                areas.update({area: cnt})
            sec = areas[max(areas.keys())]
        except ValueError as msg:
            sec = 1.0

########consider empty contours bc bad color finding setting:
        if type(sec) != np.ndarray:
            sec2 = np.array([(col,col), (0,0), (0,col), (col,0)], dtype=np.int32)
            val2 = float(col)
            dif.update({val2: [sec2,colors[col][2]]})
            #print("empty contour replaced", type(sec), colors[col][2])
        else:
            val = cv2.contourArea(sec)
            dif.update({val:[sec,colors[col][2]]})

########save the maximum contour, move a little from border by adding 100 (estimate):
    maxcont = max(dif.keys())
    cont = dif[maxcont][0]
    grade = dif[maxcont][1]
    #print(type(grade))


########Draw found contour as filled space onto black image that is big enough for pic to turn (700,700):
    blank = np.zeros(img.shape, np.uint8) #black image

    #imagergb = cv2.drawContours(img.copy(), cont, -1, (255, 255, 255), 2)
    imagergb = cv2.fillPoly(blank, pts=[cont], color=(0,0,255)) #white colour
    finalimage = cv2.copyMakeBorder(imagergb,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
    #cv2.imshow("finder", imagergb); cv2.waitKey(0)

########Draw fill the contour
    return finalimage, grade

'''Get contour function: 
takes the prprocessed RGB image and returns a tupel with 
0 beeing the image itself and 1 beeing the shape of the image as contour
Input: Image(red&black)
Output: tuple-image(red&black),Contour(corresponding)'''
def getcontour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grays out the image
    # Find contours
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    return (image,c)

'''Convex Hull function: 
takes the preprocessed image and changes the shape of the strawberry into a convex form and 
returns an image in black and red of that new form
Input: image(red&black)
Output: Image(red&black,"convex")'''
def convexHull(set):
    hull = cv2.convexHull(set[1])
    newimage = cv2.fillPoly(set[0].copy(), pts=[hull], color=(0,0,255))
    return newimage

'''Center of Mass function:
Function that takes an image that is only red (255,0,0) and black (0,0,0) and returns the x and y coordinate
(y=608, x=504, depth=3) of the center of mass based, so that the number of pixels above and below in both x and y are the same
Input: Image(Red&black)
Output: tuple-X&Y from center of mass'''
def getcenterofmass(image):
    #Convert to grayscale
    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Threshold via Otsu + bias adjustment:
    threshValue, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply an erosion + dilation to get rid of small noise:

    # Set kernel (structuring element) size:
    kernelSize = 3

    # Set operation iterations:
    opIterations = 3

    # Get the structuring element:
    maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

    # Perform closing:
    openingImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, maxKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
    openingImage = cv2.bitwise_not(openingImage)

    # Calculate the moments
    imageMoments = cv2.moments(openingImage)

    # Compute centroid, if no centroid given, take center of image
    #cv2.imshow("centerpic", image); cv2.waitKey(0)
    try:
        cx = int(imageMoments['m10'] / imageMoments['m00'])
        cy = int(imageMoments['m01'] / imageMoments['m00'])
    except ZeroDivisionError as msg:
        cx = image.shape[1]//2
        cy = image.shape[0]//2
        print(msg)

    # return points
    return (cx,cy)

'''Find lowest Extrempoint Function:
Function takes a set of image and corresponding contour and returns lowest point of the contour
Input: tuple-image(red&black), Contour(corresponding)
Output: tuple-X&Y from lowest point red'''
def findlowestpoint(set):
# Obtain outer coordinates
    c = set[1]
    bottom = tuple(c[c[:, :, 1].argmax()][0])
    return bottom

'''Rotate to lowest Function:
Function takes an image and the corresponding center of mass and Lowest point and rotates it until both of the points are
aligned vertically.
Input: (Image, tuple-centerofmass, tuple-lowestpoint)
Output: (Image, int(angle))'''
def rotate_image(image, centerofmass, lowestpoint):
####Calculate Vectors from Center of Mass to top and from Center of Mass to lowest point
    vector1 = [0, 0 - list(centerofmass)[1]]
    vector2 = [list(lowestpoint)[0]-list(centerofmass)[0], list(lowestpoint)[1]-list(centerofmass)[1]]

####Unify vectors to use dotproduct
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2) #calculates the shortest angle -> angle will always be <0

####transfer Rad into Deg and get the right angle, angle < 0 for left and angle > 0 for right turns
    angle = (1 - np.arccos(dot_product) / pi) * 180
    if vector2[0] >= 0: #lowest point on RIGHT side of center of mass
        angle *= -1
    else:#lowest point on LEFT side of center of mass
        pass

####get rotation Matrix and perform turn
    rot_mat = cv2.getRotationMatrix2D(centerofmass, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

####Ouputs an image that is turned
    return result, angle

def rotimgtolowest(image):
    '''Function takes a preprocessed image (red and black, convexHull) and turns back and forth until the "pointy" end is
       the lowest point. Used Functions: rotate_image, findlowestpoint, getcenterofmass
        Input: Image(red&black, "convex")
        Output: Image(red&black, "convex-turned"'''
    stopp = 1
    rotimg = image.copy()
    while stopp <= 3:
        # get center of mass and lowest point:
        centerofmass = getcenterofmass(rotimg)
        step1 = getcontour(rotimg)
        lowestpoint = findlowestpoint(step1)

        # draw both points:
        #step2 = cv2.circle(rotimg, centerofmass, 2, (0, 255, 255), -1)
        #rotimg = cv2.circle(step2, lowestpoint, 2, (255,255,0), -1)

        # turn picture so that center of mass and lowest point align vertically
        step4 = rotate_image(rotimg, centerofmass, lowestpoint)
        stopp += 1
        if int(step4[1]) == 0: break
        else: rotimg = step4[0]

    # return the rotated image
    return step4[0]

'''Volume Function: function calls function
Function takes an image, calculates the center of mass, the distance between it and the center of the two parts
left and right of the main center of mass and the Area of red particels in each half. Then using the first rule of 
Guldini to create a rotation Volume by rotating both sides for pi. Resulting in an float value for the Volume in 
cubig pixels
Input: Image(red&black, "vertically aligned")
Output: Float-Volume in cubic pixels
'''
def calculatemass(image):
    center = getcenterofmass(image)[0]

    #cut image in half where center of mass is
    imageright = image[:,center:]
    imageleft = image[:,:center]

    # calculate the distance between center of mass of right and left side relatif to main center of mass
    centerright = getcenterofmass(imageright)[0]
    centerleft = getcenterofmass(imageleft)[0]
    distright = centerright * pix
    distleft = (center - centerleft) * pix

    #calculate area of both sides
    arearight = countpixels(imageright) * (pix ** 2)
    arealeft = countpixels(imageleft) * (pix ** 2)

    # calculate first volume of right and left, if one is very small (under 10%), calculate all with big side
    if arealeft/(arealeft + arearight) <= 0.1:
        volumeleft = pi * distright * arearight
    else: volumeleft = pi * distleft * arealeft

    if arearight/(arealeft + arearight) <= 0.1:
        volumeright = pi * distright * arealeft
    else: volumeright = pi * distright * arearight


    mass = (volumeright + volumeleft)*dens

    return mass

def countpixels(image):
    #calculate area of both sides
        #Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #invert white and black
    invert = cv2.bitwise_not(gray)

        #calculate area
    area = cv2.countNonZero(invert)

    return area

'''Process all steps function:
Fnction that takes one image and processes it: first finds a shape, then makes the shape convex, then turns it and 
finds is volume and mass
Input: image(unprocessed), list of lists-color constants
Output: tuple of tuples-(original image, first step, scond step, thrid step),(mass)'''
def processallsteps(image,colors):
    # convert image into rgb in red and black
    start = getrgbcontourimage(image, colors)
    first = start[0]
    grade = start[1]

    # change contour of rgb image into convexHull image
    sec = getcontour(first)
    third = convexHull(sec)

    # test rotatetolowest function:
    forth = rotimgtolowest(third)

    # calculate Volume and print it out
    #print(count, calculatemass(forth), "Mass in g", sep="\t")

    # display all images
    image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=0)

    return ((image,first,third,forth),calculatemass(forth),grade)

'''find best picture function:
Function detects the best picture by calculating the difference between the original shape and the convex hull, the one
with the least is the best picture
Input: folderpath and the foldername as strings and list of list (colors)
Output: Image (red&black) or tuple of images or mass or all depending on indices 
'''
def bestpic(folderpath, folder1,colors):
    dif1 = 10000
    # find pictures in corresponding folder: only pictures of one Strawbery
    folders = os.listdir(folderpath)
    lof = []
    for folder in folders:
        if folder == folder1:
            path = folderpath + folder + "\\"
            break

    # load all the files in the right folder
    files = os.listdir(path)
    for f in files:
        lof.append(path + f)

    #load picture files to get started with processing
    for file in lof:
        if file[-3:] != "jpg":  # check if file is a picture
            pass
        else:
            image = cv2.imread(file)
            #print(file)

    #use all the processing functions and save accordingly
            first = getrgbcontourimage(image, colors)[0]

    # calculate area of raw found shape
            arearaw = countpixels(first)

    # calculate area of the convex hull picture
            sec = getcontour(first)
            third = convexHull(sec)
            areaconvex = countpixels(third)

    # calculate difference
            dif = areaconvex - arearaw

    # save if lowest dif
            if dif < dif1:
                # [0][0]...for original, [0][1]...for first, [0][2]...for convex, [0][3]...for turned, [1][0]... for mass as integer
                best = image
                dif1 = dif

    return processallsteps(image,colors)


'''Grade strawberies Function: under construction
Function takes the processed data of the best picture and uses the convex picture to create a mask and lay it over the 
original image. It then uses predefined constants, 5 in total, to distinguish the ripeness
1...white color in the picture -> still a flower
2...dominant green in the picture -> still a green Strawberry
3...dominant light red in the picture -> almost ready Strawberry
4...dominant red in the picture -> perfect Strawberry
5...dominant darker red in the picture -> overly ripe Strawberry
Input: tuple of tuples: ((img,image,image,image),float)
Output: integer'''
def grade(imagetuple):
    # colorconstants: list of lists containing: 0_colorconstant1, 1_colorconstant2, 2_number of state, 3_description of state
    colorconstants = [
          [np.array([0,50,55]),np.array([13,255,255]), 1,"still a Flower"],     #color found from testenv.py using colors.jpg
          [np.array([0,121,67]),np.array([7,255,255]), 2, "still green"],    #color found from testenv.py using right000005.jpg
          [np.array([0,189,105]),np.array([20,255,255]), 3, "almost ready"],  #color found from testenv.py using right0004065.jpg
          [np.array([0,205,69]),np.array([179,255,255]), 4, "perfect Strawberry"],  #color found from testenv.py using right0004065.jpg
          [np.array([14,92,119]),np.array([35,200,255]), 5, "your old man"],  #color found from testenv.py using pics in StrawberyNotReady folder
          #[np.array([0,34,67]),np.array([56,255,255]), 6, "green1"]   #colour found from testenv.py using right0001963.jpg
          ]
    area1 = 1.
    code = (0,0)

    # get the contour and original image:
    conveximage = imagetuple[0][2].copy()
    contour = getcontour(conveximage)[1]
    original = imagetuple[0][0].copy()

    # create mask of contour and only show part of original picture in that mask
    stencil = np.zeros(original.shape).astype(original.dtype)
    stencil = cv2.fillPoly(stencil, pts=[contour], color=(255, 255, 255))
    result = cv2.bitwise_and(original, stencil)
    #cv2.imshow("scoped", result); cv2.waitKey(0)

    # create hsv plane and check for constants
    for col in colorconstants:
        hsvimage = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        blurhsv = cv2.GaussianBlur(hsvimage, (5, 5), 0)  # optional, does not do that much, initial (5,5)
        mask = cv2.inRange(blurhsv, col[0], col[1])
        edgeimage = cv2.Canny(mask, 50, 100, 20)
        kernel = np.ones((2, 2))  # initial 5,5 smaler kernel works better with smaller pictures
        dialimage = cv2.dilate(edgeimage, kernel, iterations=2)  # initial 2
        first = cv2.erode(dialimage, kernel, iterations=1)  # initial 1
        #cv2.imshow("findings", first); cv2.waitKey(0)


    # find maximum contour of this colour setting:
        contours, hierachy = cv2.findContours(first, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            c = max(contours, key=cv2.contourArea)
        except ValueError:
            c = np.array([(1, 1), (0, 0), (0, 1), (1, 0)], dtype=np.int32)

    # for the maximum area save grading code 1 to 5
        area = cv2.contourArea(c)
        if area > area1:
            area1 = area
            code = (col[2],col[3])

    # print message of flowertype
    #print(code)

    #retrun just the grading number
    return code[0]

#display functions: Functions that are used to display certain steps of the developement
def preprocessimghsv(img,colours):
    """Tests which colour setting in my colors is the best to find contours"""
    hsvimage = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    blurhsv = cv2.GaussianBlur(hsvimage, (5, 5), 0)  # optional, does not do that much, initial (5,5)
    for colour in colours:
        print(colour[2])
        mask = cv2.inRange(blurhsv, colour[0], colour[1])
        cv2.imshow("Preprocessed Image", mask)
        cv2.waitKey(0)

def displayall(path, colors):
    #load files
    samples = getpicspath(path)
    count = 0

    grades = {0:0,1:0,2:0,3:0,4:0,5:0}
    gradesfunc = grades.copy()

    # create image files if pictures:
    for file in samples:
        count += 1
        if file[-3:] != "jpg":  # check if file is a picture
            pass
        elif count != 0:
            image = cv2.imread(file)
            # process all steps
            steps = processallsteps(image, colors)
            print(file,"Mass: ", steps[1], "Grade: ", steps[2], sep="\t")

            g = grade(steps)
            gradesfunc.update({g:gradesfunc[g]+1})
            grades.update({steps[2]:grades[steps[2]]+1})


            #display all steps
            cv2.imshow("original -> schape -> convexShape -> turned", np.hstack(steps[0])); cv2.waitKey(100)
    print(grades, gradesfunc, sep="\n")

# Get data function: Functions that help to get the data
def getpicspath(folderpath):
    '''Function lets one display all the pictures in a certain folder'''
    folders = os.listdir(folderpath)
    listoffolders = []
    lof = []
    for folder in folders:
        listoffolders.append(folderpath + folder + "\\")
        #if folder == "5_groseErdbeere" or folder == "6_shraegeErdbeere": listoffolders.append(folderpath+folder+"\\")
        #if folder == "Strawberry" or folder == "StrawberryNotReady": listoffolders.append(folderpath + folder + "\\")

    for i in listoffolders:
        files = os.listdir(i)
        for f in files:
            lof.append(i+f)
    return lof

'''Display all images and use the functions:'''
if __name__ == '__main__':

    displayall(pictures,mycolors) #to display all images in strawberry and strawberrynotready folder

    pfad = "C:\\Users\\gerwi\\Dropbox\\uni\\BA_Erdbeeren\\Datasets\\exp\\hole examples\\"
    folders = os.listdir(pfad)
    for f in folders:
        if f != "Flower" and f != "StrawberryNotReady" and f != "Strawberry":
            tupleimage = bestpic(pfad,f,mycolors)
            #cv2.imshow("best5", np.hstack(tupleimages[0])); cv2.waitKey(0)
            #mask = grade(tupleimage)
            print("Folder: " + f, "Mass: " + str(tupleimage[1]), "Grade: " + str(tupleimage[2]), sep="\t")
            cv2.imshow("best5", np.hstack(tupleimage[0])); cv2.waitKey(0)





