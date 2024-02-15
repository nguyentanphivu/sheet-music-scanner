import numpy as np
import cv2
from copy import deepcopy

def blackHorizontalLinesRowInd(imgGray, imgH, imgW, threshold):

    numBlackPixel = int(threshold * imgW)
    numWhitePixel = imgW - numBlackPixel
    totalThreshold = numWhitePixel * 255

    rowsWithLine = []

    for row in range(imgH):

        curRow = imgGray[row][:]
        total = np.sum(curRow)

        if (total < totalThreshold):
            rowsWithLine.append(row)

    return rowsWithLine

def trimRowsWithLine(rowsWithLine, minGap = 5):
    
    trimmedRowsWithLine = []
    trimmedRowsWithLine.append(rowsWithLine[0])

    for i in range(1, len(rowsWithLine)):
        if ((rowsWithLine[i] - rowsWithLine[i - 1]) > minGap):
            trimmedRowsWithLine.append(rowsWithLine[i])
        
    return trimmedRowsWithLine

def linesClustering(trimmedRowsWithLine, minGap = 50):
    clusterLst = []

    cluster = []
    cluster.append(trimmedRowsWithLine[0])
    for i in range(1, len(trimmedRowsWithLine)):

        if ((trimmedRowsWithLine[i] - trimmedRowsWithLine[i - 1]) > minGap):
            clusterLst.append(cluster)
            cluster = []
            cluster.append(trimmedRowsWithLine[i])

        else:
            cluster.append(trimmedRowsWithLine[i])
    clusterLst.append(cluster)

    return clusterLst

def blackVerticalLinesColumnInd(imgGray, imgH, imgW, clusterLst, threshold):

    columnsWithLine = dict()

    for cluster in clusterLst:

        clusterRowBegin = cluster[0]
        clusterRowEnd = cluster[4]
        columnsWithLine[(clusterRowBegin, clusterRowEnd)] = []

        imgROI = imgGray[clusterRowBegin: clusterRowEnd][:]
        imgROIHeight = clusterRowEnd - clusterRowBegin

        numBlackPixel = int(threshold * imgROIHeight)
        numWhitePixel = imgROIHeight - numBlackPixel
        totalThreshold = numWhitePixel * 255

        for column in range(imgW):

            curCol = imgROI[:, column]
            total = np.sum(curCol)

            if (total < totalThreshold):
                columnsWithLine[(clusterRowBegin, clusterRowEnd)].append(column)
    
    return columnsWithLine
                
def trimColumnsWithLine(columnsWithLine, minGap = 100):
    
    trimmedColumnsWithLine = dict()

    for key in columnsWithLine.keys():
        trimmedColumnsWithLine[key] = []
        value = columnsWithLine[key]
        trimmedColumnsWithLine[key].append(value[0])

        for i in range(1, len(columnsWithLine[key])):
            if ((value[i] - value[i - 1]) > minGap):
                trimmedColumnsWithLine[key].append(value[i])
            
    return trimmedColumnsWithLine

def averageLinesGap(clusterLst):
    outpGap = 0
    for cluster in clusterLst:
        clusterRowBegin = cluster[0]
        clusterRowEnd = cluster[4]
        gap = (clusterRowEnd - clusterRowBegin) / 4
        outpGap += gap
    
    return outpGap/len(clusterLst)

def matchTemplateRemake(img, template, threshold, drawBoundingBox = True, suppression = False):

    templateH, templateW = template.shape

    outpConfidence = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    boundingBoxUpperLeftCoor = np.where(outpConfidence > threshold)

    xCoors, yCoors = list(boundingBoxUpperLeftCoor[1]), list(boundingBoxUpperLeftCoor[0])
    combineXYCoors = list(zip(xCoors, yCoors))
    sortedCombine = list(zip(*sorted(combineXYCoors)))

    if (len(sortedCombine) == 0):
        xCoorsSorted = []
        yCoorsSorted = []
    else:
        xCoorsSorted, yCoorsSorted = list(sortedCombine[0]), list(sortedCombine[1])
    boundingBoxUpperLeftCoor = (np.array(xCoorsSorted), np.array(yCoorsSorted))

    if suppression:
        print('Before:', boundingBoxUpperLeftCoor, boundingBoxUpperLeftCoor[0].shape)
        boundingBoxUpperLeftCoor = boundingBoxSuppression(boundingBoxUpperLeftCoor)
        print('After:', boundingBoxUpperLeftCoor, boundingBoxUpperLeftCoor[0].shape)

    if drawBoundingBox:

        imgCopy = deepcopy(img)
        imgCopy = cv2.cvtColor(imgCopy, cv2.COLOR_GRAY2BGR)
        for pt in zip(*boundingBoxUpperLeftCoor):
            #draw a rectangle where it exceeds the threshold
            cv2.rectangle(imgCopy, pt, (pt[0] + templateW, pt[1] + templateH), (0, 255, 0), 2)

        return boundingBoxUpperLeftCoor, outpConfidence, imgCopy

    return boundingBoxUpperLeftCoor, outpConfidence, None

def boundingBoxSuppression(matchedCoordinates, threshold = 10):

    if ((len(matchedCoordinates[0]) == 0) or (len(matchedCoordinates[0]) == 1)):
        return matchedCoordinates

    matchedCoordinatesMatrix = np.stack((matchedCoordinates[0], matchedCoordinates[1])).T
    numberFounded = matchedCoordinatesMatrix.shape[0]
    distanceMatrix = np.sqrt(np.add(np.add(-2*(matchedCoordinatesMatrix@((matchedCoordinatesMatrix).T)), np.reshape(np.sum(np.square(matchedCoordinatesMatrix), axis = 1), (numberFounded, -1))), np.sum(np.square((matchedCoordinatesMatrix).T), axis = 0)))
    upperTriDistMat = np.triu(distanceMatrix)

    nearPairs = np.where((upperTriDistMat > 0) & (upperTriDistMat <= threshold))
    if (len(nearPairs[0]) == 0):
        return matchedCoordinates
    
    point1, point2 = nearPairs[0], nearPairs[1]
    
    clusters = [[point1[0]]]
    prevPoint = point1[0]

    while (len(point1) != 0):
        if (point1[0] == prevPoint):
            clusters[-1].append(point2[0])
            delInd = np.where((point1 == point2[0]))
            point1 = np.delete(point1, delInd)
            point2 = np.delete(point2, delInd)
            point1 = np.delete(point1, 0)
            point2 = np.delete(point2, 0)
        elif (point1[0] != prevPoint):
            prevPoint = point1[0]
            clusters.append([point1[0], point2[0]])
            delInd = np.where((point1 == point2[0]))
            point1 = np.delete(point1, delInd)
            point2 = np.delete(point2, delInd)
            point1 = np.delete(point1, 0)
            point2 = np.delete(point2, 0)

    trimmedMatchedCoordinates = {'x': [], 'y': []}
    delInd = np.array([])
    for cluster in clusters:
        xCoor = int(np.mean(matchedCoordinates[0][cluster]))
        yCoor = int(np.mean(matchedCoordinates[1][cluster]))
        trimmedMatchedCoordinates['x'].append(xCoor)
        trimmedMatchedCoordinates['y'].append(yCoor)
        delInd = np.concatenate((delInd, cluster))

    trimmedMatchedCoordinates['x'] = np.array(trimmedMatchedCoordinates['x'])
    trimmedMatchedCoordinates['y'] = np.array(trimmedMatchedCoordinates['y'])

    ind = np.array([i for i in range(numberFounded)])
    ind = np.delete(ind, delInd.astype(np.int64))
    trimmedMatchedCoordinates['x'] = np.concatenate((trimmedMatchedCoordinates['x'], matchedCoordinates[0][ind]))
    trimmedMatchedCoordinates['y'] = np.concatenate((trimmedMatchedCoordinates['y'], matchedCoordinates[1][ind]))

    
    return (trimmedMatchedCoordinates['x'], trimmedMatchedCoordinates['y'])

    # trimmedMatchedCoordinates = {'x': [], 'y': []}

    # for count, (x, y) in enumerate(zip(*matchedCoordinates)):

    #     if (count == 0):
    #         trimmedMatchedCoordinates['x'].append(x)
    #         trimmedMatchedCoordinates['y'].append(y)

    #     else:
    #         xDiff = x - trimmedMatchedCoordinates['x'][-1]
    #         yDiff = y - trimmedMatchedCoordinates['y'][-1]

    #         if ((abs(yDiff) > threshold) or (xDiff > threshold)):
    #             trimmedMatchedCoordinates['x'].append(x)
    #             trimmedMatchedCoordinates['y'].append(y)
    #         else:
    #             pass

    #         # # if (abs(x - trimmedMatchedCoordinates['x'][-1]) > threshold) and (abs(y - trimmedMatchedCoordinates['y'][-1]) > threshold):
    #         # # if (abs(x - trimmedMatchedCoordinates['x'][-1]) > threshold) or (abs(y - trimmedMatchedCoordinates['y'][-1]) > threshold):
    #         # # if (abs(x - trimmedMatchedCoordinates['x'][-1]) > threshold):
    #         #     trimmedMatchedCoordinates['x'].append(x)
    #         #     trimmedMatchedCoordinates['y'].append(y)
    #         # else:
    #         #     pass
    
    # tempCoordinates = (np.array(trimmedMatchedCoordinates['y']), np.array(trimmedMatchedCoordinates['x']))
    
    # outpCoordinates = {'x': [], 'y': []}

    # xCoors, yCoors = list(tempCoordinates[1]), list(tempCoordinates[0])
    # combineXYCoors = list(zip(yCoors, xCoors))
    # sortedCombine = list(zip(*sorted(combineXYCoors)))

    # if (len(sortedCombine) == 0):
    #     xCoorsSorted = []
    #     yCoorsSorted = []
    # else:
    #     xCoorsSorted, yCoorsSorted = list(sortedCombine[1]), list(sortedCombine[0])
   
    # tempMatchCoordinates = (np.array(xCoorsSorted), np.array(yCoorsSorted))

    # for count, (x, y) in enumerate(zip(*tempMatchCoordinates)):

    #     if (count == 0):
    #         outpCoordinates['x'].append(x)
    #         outpCoordinates['y'].append(y)

    #     else:

    #         xDiff = x - outpCoordinates['x'][-1]
    #         yDiff = y - outpCoordinates['y'][-1]

    #         if ((abs(xDiff) > threshold) or (yDiff > threshold)):
    #             outpCoordinates['x'].append(x)
    #             outpCoordinates['y'].append(y)
    #         else:
    #             pass
        
    #         # # if (abs(x - outpCoordinates['x'][-1]) > threshold) and (abs(y - outpCoordinates['y'][-1]) > threshold):
    #         # # if (abs(y - outpCoordinates['y'][-1]) > threshold):
    #         #     outpCoordinates['x'].append(x)
    #         #     outpCoordinates['y'].append(y)
    #         # else:
    #         #     pass

    # return (np.array(outpCoordinates['x']), np.array(outpCoordinates['y']))

def tryTemplateMatchNote(img, blackNoteTemplate, ratioWH, threshold, avgLinesGap, error = 3):

    maxMatched = -1
    chosenSize = None

    for err in range(-error, error + 1, 1):

        blackNoteH = int(avgLinesGap) + err
        blackNoteW = int(blackNoteH * ratioWH)
        blackNoteTemplateResize = cv2.resize(blackNoteTemplate, (blackNoteW, blackNoteH))
        upperLeftCornerCoor, _1, _2 = matchTemplateRemake(img, blackNoteTemplateResize, threshold, False)
        upperLeftCornerCoorTrimmed = boundingBoxSuppression(upperLeftCornerCoor)
        numMatched = upperLeftCornerCoorTrimmed[0].shape[0]

        if (numMatched > maxMatched):
            maxMatched = numMatched
            chosenSize = (blackNoteH, blackNoteW)

    return chosenSize, maxMatched

def drawHorizontalLines(rowsWithLine, imgH, imgW):
    outputImg = np.zeros((imgH, imgW))
    outputImg += 255

    for row in rowsWithLine:
        cv2.line(outputImg, (0, row), (imgW, row), (0, 0, 0), 2)

    return outputImg
