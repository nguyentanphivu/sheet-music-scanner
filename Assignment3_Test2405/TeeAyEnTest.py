import cv2
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from utils import *
from constants import *
import sys
import math

np.set_printoptions(threshold=sys.maxsize)

outputFolder = r"C:\Users\nguye\Downloads\out"
outputTextFile = open(outputFolder + r"\test.txt", 'w+')

imgPath = r"C:\Users\nguye\Downloads\346169806_1178928896107035_3916678363840706614_n1.png"
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

retval, imgFindLine = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
# retval, imgFindLine = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
# img = imgFindLine

cv2.imwrite(outputFolder + r"\originalImage.jpg", img)
cv2.imwrite(outputFolder + r"\thresholdImg.jpg", imgFindLine)

HEIGHT, WIDTH = img.shape

rowsWithLine = blackHorizontalLinesRowInd(imgFindLine, HEIGHT, WIDTH, 0.35)
trimmedRowsWithLine = trimRowsWithLine(rowsWithLine, minGap = 3)

cv2.imwrite(outputFolder + r"\horizontalLine.jpg", drawHorizontalLines(trimmedRowsWithLine, img))

lineClusterLst = linesClustering(trimmedRowsWithLine, minGap = 50)
avgLinesGapCluster = averageLinesGapCluster(lineClusterLst)
avgLinesGap = averageLinesGapAll(avgLinesGapCluster)


# Print
print('Original rows with line:', rowsWithLine)
print('Trimmed rows with line:', trimmedRowsWithLine)
print('There are', len(trimmedRowsWithLine), 'lines')
print('There are', len(lineClusterLst), 'line clusters')
print('Line clusters: %s' % str(lineClusterLst))
print('Average line gap cluster:', avgLinesGapCluster)
print('Average line gap:', avgLinesGap)
outputTextFile.write('Original rows with line: %s \r\n' % str(rowsWithLine))
outputTextFile.write('Trimmed rows with line: %s \r\n' % str(trimmedRowsWithLine))
outputTextFile.write('There are %d lines \r\n' % len(trimmedRowsWithLine))
outputTextFile.write('There are %d line clusters \r\n' % len(lineClusterLst))
outputTextFile.write('Line cluster list: %s \r\n' % str(lineClusterLst))
outputTextFile.write('Average line gap cluster: %s \r\n' % str(avgLinesGapCluster))
outputTextFile.write('Average line gap: %d \r\n' % avgLinesGap)


columnsWithLine = blackVerticalLinesColumnInd(imgFindLine, HEIGHT, WIDTH, lineClusterLst, threshold = 0.99)
trimmedColumnsWithLine = trimColumnsWithLine(columnsWithLine, minGap = 100)

# Print
print('Original columns with line: %s' % str(columnsWithLine))
print('Trimmed columns with line %s' % str(trimmedColumnsWithLine))
outputTextFile.write('Original columns with line: %s \r\n' % str(columnsWithLine))
outputTextFile.write('Trimmed columns with line: %s \r\n' % str(trimmedColumnsWithLine))

noCluster = len(lineClusterLst)
boundLst = list()
finalOutputPitch = []
finalOutputDuration = []

if (noCluster == 1):
    cluster = lineClusterLst[0]
    uBound = max(int(cluster[0] - (3 * avgLinesGap)), 0)
    lBound = min(int(cluster[-1] + (3 * avgLinesGap)), HEIGHT)
    bound = (uBound, lBound)
    boundLst.append(bound)
else:
    lBound = lineClusterLst[0][-1] + int((lineClusterLst[1][0] - lineClusterLst[0][-1])/2)
    uBound = max(lineClusterLst[0][0] - lBound + lineClusterLst[0][-1], 0)
    boundLst.append((uBound, lBound))

    for clusterID in range(1, noCluster - 1):

        cluster = lineClusterLst[clusterID]
        uBound = boundLst[-1][1]
        lBound = cluster[-1] + int((lineClusterLst[clusterID + 1][0] - cluster[-1])/2)
        boundLst.append((uBound, lBound))

    uBound = boundLst[-1][1]
    lBound = min(lineClusterLst[-1][-1] + uBound - lineClusterLst[-2][-1], HEIGHT)
    boundLst.append((uBound, lBound))

# Save
ROILines = []
for bound in boundLst:
    ROILines.append(bound[0])
ROILines.append(boundLst[-1][1])
ROILinesImg = drawHorizontalLines(ROILines, img)
cv2.imwrite(outputFolder + r"\ROIImage.jpg", ROILinesImg)

count = 0
for bound in boundLst:
    upperBound, lowerBound = bound
    ROI = deepcopy(img[upperBound : lowerBound, :])
    cv2.imwrite(outputFolder + r"\cluster" + str(count) + r".jpg", ROI)
    lastLine = lineClusterLst[count][-1] - upperBound
    cv2.imwrite(outputFolder + r"\cluster" + str(count) + r"LastLine.jpg", drawHorizontalLines([lastLine], ROI))

    clusterBegin, clusterEnd = list(avgLinesGapCluster)[count]
    useAvgLinesGap = avgLinesGapCluster[(clusterBegin, clusterEnd)]
            
    CLEF_H = int(useAvgLinesGap * CLEF_H_LINE_GAP_RATIO)
    CLEF_W = int(useAvgLinesGap * CLEF_H_LINE_GAP_RATIO * CLEF_W_H_RATIO)
    clefLoc, clefRes, clefBox = findClef(ROI, clefTemplate, CLEF_H, CLEF_W, outputTextFile)
    cv2.imwrite(outputFolder + r"\cluster" + str(count) + r"ClefBox.jpg", clefBox)

    # if (clefLoc[0].shape[0] > 1):
    if (clefLoc[0].shape[0] != 1):
        print('Invalid: There are %d clef(s) founded' % clefLoc[0].shape[0])
        outputTextFile.write('Invalid: There are %d clef(s) founded for cluster %d \r\n' % (clefLoc[0].shape[0], count + 1))
        #break
    else:
        ROI = deepcopy(ROI[:, clefLoc[0][0] + CLEF_W : ])

    blackLoc, blackRes, blackBox, blackChosenSize = findBlackNote(ROI, blackNoteTemplate, 0.9, 0.67, useAvgLinesGap, 1, outputTextFile)
    cv2.imwrite(outputFolder + r"\cluster" + str(count) + r"BlackBox.jpg", blackBox)

    whiteLoc, whiteRes, whiteBox, whiteChosenSize = findWhiteNote(ROI, whiteNoteTemplate, 0.7, 0.7, useAvgLinesGap, 1, outputTextFile)
    cv2.imwrite(outputFolder + r"\cluster" + str(count) + r"WhiteBox.jpg", whiteBox)

    whiteAloneLoc, whiteAloneRes, whiteAloneBox, whiteAloneChosenSize = findWhiteAloneNote(ROI, whiteAloneNoteTemplate, 0.7, 0.7, useAvgLinesGap, 1, outputTextFile)
    cv2.imwrite(outputFolder + r"\cluster" + str(count) + r"WhiteAloneBox.jpg", whiteAloneBox)

    hookLoc, hookRes, hookBox, hookChosenSize = findHook(ROI, hookTemplate, 0.6, useAvgLinesGap, outputTextFile)
    cv2.imwrite(outputFolder + r"\cluster" + str(count) + r"HookBox.jpg", hookBox)

    hookRotatedLoc, hookRotatedRes, hookRotatedBox, hookRotatedChosenSize = findHookRotated(ROI, hookRotatedTemplate, 0.6, useAvgLinesGap, outputTextFile)
    cv2.imwrite(outputFolder + r"\cluster" + str(count) + r"HookRotatedBox.jpg", hookRotatedBox)

    wholeRestLoc, wholeRestRest, wholeRestBox = findRests(ROI, wholeRestTemplate, W_H_RATIO_WHOLE_REST, 'Whole Rest', 0.8, useAvgLinesGap, outputTextFile)
    cv2.imwrite(outputFolder + r"\cluster" + str(count) + r"WholeRestBox.jpg", wholeRestBox)

    halfRestLoc, halfRestRest, halfRestBox = findRests(ROI, halfRestTemplate, W_H_RATIO_HALF_REST, 'Half Rest', 0.72, useAvgLinesGap, outputTextFile)
    cv2.imwrite(outputFolder + r"\cluster" + str(count) + r"HalfRest.jpg", halfRestBox)

    quarterRestLoc, quarterRestRest, quarterRestBox = findRests(ROI, quarterRestTemplate, W_H_RATIO_QUARTER_REST, 'Quarter Rest', 0.6, useAvgLinesGap, outputTextFile)
    cv2.imwrite(outputFolder + r"\cluster" + str(count) + r"QuarterRest.jpg", quarterRestBox)

    eighthRestLoc, eighthRestRest, eighthRestBox = findRests(ROI, eighthRestTemplate, W_H_RATIO_EIGHTH_REST, 'Eighth Rest', 0.8, useAvgLinesGap, outputTextFile)
    cv2.imwrite(outputFolder + r"\cluster" + str(count) + r"EighthRest.jpg", eighthRestBox)

    blackPitch = deepcopy(blackLoc[1])
    whitePitch = deepcopy(whiteLoc[1])
    whiteAlonePitch = deepcopy(whiteAloneLoc[1])

    blackPitch = blackPitch + int(blackChosenSize[1]/2)
    whitePitch = whitePitch + int(whiteChosenSize[1]/2)
    whiteAlonePitch = whiteAlonePitch + int(whiteAloneChosenSize[1]/2)

    blackPitch = - (blackPitch - lastLine)
    whitePitch = - (whitePitch - lastLine)
    whiteAlonePitch = - (whiteAlonePitch - lastLine)

    blackPitch = blackPitch / useAvgLinesGap
    whitePitch = whitePitch / useAvgLinesGap
    whiteAlonePitch = whiteAlonePitch / useAvgLinesGap

    blackPitch = blackPitch / (1/2)
    whitePitch = whitePitch / (1/2)
    whiteAlonePitch = whiteAlonePitch / (1/2)

    blackPitch = (np.around(blackPitch)).astype(int)
    whitePitch = (np.around(whitePitch)).astype(int)
    whiteAlonePitch = (np.around(whiteAlonePitch)).astype(int)

    blackPitch = blackPitch + 2
    whitePitch = whitePitch + 2
    whiteAlonePitch = whiteAlonePitch + 2

    blackDuration = np.full_like(blackPitch, 4)
    whiteDuration = np.full_like(whitePitch, 2)
    whiteAloneDuration = np.full_like(whiteAlonePitch, 1)
    wholeRestDuration = np.full_like(wholeRestLoc[0], 1)
    halfRestDuration = np.full_like(halfRestLoc[0], 2)
    quarterRestDuration = np.full_like(quarterRestLoc[0], 4)
    eighthRestDuration = np.full_like(eighthRestLoc[0], 8)

    pitchOut = {'black': [], 'white': [], 'whiteAlone': []}
    originalPitch = 4

    for pitch in blackPitch:
        if ((0 <= pitch) and (pitch <= 6)):
            outpPitch = noteList[pitch] + str(originalPitch)
            pitchOut['black'].append(outpPitch)
        elif (pitch < 0):
            pitchNormalized = pitch + (7 * math.ceil(abs(pitch) / 7))
            outpPitch = noteList[pitchNormalized] + str(originalPitch - math.ceil(abs(pitch) / 7))
            pitchOut['black'].append(outpPitch)
        elif (pitch > 6):
            pitchNormalized = pitch - (7 * math.ceil(pitch / 7))
            outpPitch = noteList[pitchNormalized] + str(originalPitch + math.floor(abs(pitch) / 7))
            # outputTextFile.write('%s' %str((abs(pitch) / 7)))
            pitchOut['black'].append(outpPitch)
    print('Cluster %d black pitch: %s' % (count, str(pitchOut['black'])))
    outputTextFile.write('Cluster %d black pitch: %s \r\n' % (count, str(pitchOut['black'])))

    for pitch in whitePitch:
        if ((0 <= pitch) and (pitch <= 6)):
            outpPitch = noteList[pitch] + str(originalPitch)
            pitchOut['white'].append(outpPitch)
        elif (pitch < 0):
            pitchNormalized = pitch + (7 * math.ceil(abs(pitch) / 7))
            outpPitch = noteList[pitchNormalized] + str(originalPitch - math.ceil(abs(pitch) / 7))
            pitchOut['white'].append(outpPitch)
        elif (pitch > 6):
            pitchNormalized = pitch - (7 * math.ceil(pitch / 7))
            outpPitch = noteList[pitchNormalized] + str(originalPitch + math.ceil(abs(pitch) / 7))
            pitchOut['white'].append(outpPitch)
    print('Cluster %d white pitch: %s' % (count, str(pitchOut['white'])))
    outputTextFile.write('Cluster %d white pitch: %s \r\n' % (count, str(pitchOut['white'])))

    for pitch in whiteAlonePitch:
        if ((0 <= pitch) and (pitch <= 6)):
            outpPitch = noteList[pitch] + str(originalPitch)
            pitchOut['whiteAlone'].append(outpPitch)
        elif (pitch < 0):
            pitchNormalized = pitch + (7 * math.ceil(abs(pitch) / 7))
            outpPitch = noteList[pitchNormalized] + str(originalPitch - math.ceil(abs(pitch) / 7))
            pitchOut['whiteAlone'].append(outpPitch)
        elif (pitch > 6):
            pitchNormalized = pitch - (7 * math.ceil(pitch / 7))
            outpPitch = noteList[pitchNormalized] + str(originalPitch + math.ceil(abs(pitch) / 7))
            pitchOut['whiteAlone'].append(outpPitch)
    print('Cluster %d whiteAlone pitch: %s' % (count, str(pitchOut['whiteAlone'])))
    outputTextFile.write('Cluster %d whiteAlone pitch: %s \r\n' % (count, str(pitchOut['whiteAlone'])))

    restPitch = [0] * (wholeRestLoc[0].shape[0] + halfRestLoc[0].shape[0] + quarterRestLoc[0].shape[0] + eighthRestLoc[0].shape[0])

    noteCombine = np.concatenate((blackLoc[0], whiteLoc[0], whiteAloneLoc[0], wholeRestLoc[0], halfRestLoc[0], quarterRestLoc[0], eighthRestLoc[0]))
    pitchOutCombine = pitchOut['black'] + pitchOut['white'] + pitchOut['whiteAlone'] + restPitch
    indSort = np.argsort(noteCombine)
    pitchOutCombineSorted = []
    for ind in indSort:
        pitchOutCombineSorted.append(pitchOutCombine[ind])

    print('Cluster %d pitch sorted: %s' % (count, str(pitchOutCombineSorted)))
    outputTextFile.write('Cluster %d pitch sorted: %s \r\n' % (count, str(pitchOutCombineSorted)))

    finalOutputPitch = finalOutputPitch + pitchOutCombineSorted

    for hook in hookLoc[0]:
        distance = abs(blackLoc[0] + blackChosenSize[0] - hook)
        minDistanceInd = np.argmin(distance)
        if (distance[minDistanceInd] <= 5):
            blackDuration[minDistanceInd] = 8

    for hook in hookRotatedLoc[0]:
        distance = abs(blackLoc[0] - hook)
        minDistanceInd = np.argmin(distance)
        if (distance[minDistanceInd] <= 5):
            blackDuration[minDistanceInd] = 8

    durationCombine = np.concatenate((blackDuration, whiteDuration, whiteAloneDuration, wholeRestDuration, halfRestDuration, quarterRestDuration, eighthRestDuration))
    durationCombineSorted = durationCombine[indSort]

    finalOutputDuration = np.concatenate((finalOutputDuration, durationCombineSorted))


    count += 1 

print('There are total %d notes' % len(finalOutputPitch))
outputTextFile.write('There are total %d notes' % len(finalOutputPitch))
print('There are total %d notes' % len(finalOutputDuration))
outputTextFile.write('There are total %d durations' % len(finalOutputDuration))
print('All note sorted: %s' % str(finalOutputPitch))
outputTextFile.write('All note sorted: %s \r\n' % str(finalOutputPitch))
print('All duration sorted: %s' % str(finalOutputDuration))
outputTextFile.write('All duration sorted: %s' % str(finalOutputDuration))

if (len(finalOutputDuration) != len(finalOutputPitch)):
    print('Something is wrong %d duration and %d pitch' % (len(finalOutputDuration), len(finalOutputPitch)))
    outputTextFile.write('Something is wrong %d duration and %d pitch' % (len(finalOutputDuration), len(finalOutputPitch)))
