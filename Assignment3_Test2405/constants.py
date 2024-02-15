import numpy as np
import cv2

# noteList = ['E', 'F', 'G', 'A', 'B', 'C', 'D']
noteList = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

clefTemplatePath = r"C:\Users\nguye\Downloads\templates\clef\clef.png"
clefTemplate = cv2.imread(clefTemplatePath, cv2.IMREAD_GRAYSCALE)

CLEF_W_H_RATIO = 3/8 #W/H
CLEF_H_LINE_GAP_RATIO = 7.35

blackNoteTemplate = np.array([
    [255, 255, 255, 0, 0, 0, 0, 0, 255],
    [255, 255, 0, 0, 0, 0, 0, 0, 0],
    [255, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 255],
    [0, 0, 0, 0, 0, 0, 0, 255, 255],
    [255, 0, 0, 0, 0, 0, 255, 255, 255],
], dtype = np.uint8) # Ratio W/H = 10/8 = 5/4

whiteNoteTemplate = np.array([
    [255, 255, 255, 0  , 0  , 0  , 0  , 0  , 255],
    [255, 255, 0  , 0  , 0  , 255, 255, 255, 0  ],
    [255, 0  , 0  , 0  , 255, 255, 255, 255, 0  ],
    [0  , 0  , 0  , 255, 255, 255, 255, 0  , 0  ],
    [0  , 0  , 255, 255, 255, 255, 0  , 0  , 0  ],
    [0  , 255, 255, 255, 255, 0  , 0  , 0  , 255],
    [0  , 255, 255, 255, 0  , 0  , 0  , 255, 255],
    [255, 0  , 0  , 0  , 0  , 0  , 255, 255, 255],
], dtype = np.uint8) # Ratio W/H = 10/8 = 5/4

whiteAloneNoteTemplate = np.array([
    [255, 255, 255, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 255, 255, 255],
    [255, 0  , 0  , 0  , 255, 255, 255, 255, 0  , 0  , 0  , 255, 255],
    [255, 0  , 0  , 255, 255, 255, 255, 255, 255, 0  , 0  , 0  , 255],
    [0  , 0  , 0  , 255, 255, 255, 255, 255, 255, 0  , 0  , 0  , 0  ],
    [0  , 0  , 0  , 0  , 255, 255, 255, 255, 255, 255, 0  , 0  , 0  ],
    [255, 0  , 0  , 0  , 255, 255, 255, 255, 255, 255, 0  , 0  , 255],
    [255, 255, 0  , 0  , 0  , 255, 255, 255, 255, 0  , 0  , 0  , 255],
    [255, 255, 255, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 255, 255, 255],
], dtype = np.uint8) # Ratio W/H = 13/8

# hookTemplate = np.array([
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
#     [0  , 0  , 255, 255, 255, 255, 255, 255, 255, 255, 255],
#     [0  , 0  , 255, 255, 255, 255, 255, 255, 255, 255, 255],
#     [0  , 0  , 0  , 255, 255, 255, 255, 255, 255, 255, 255],
#     [0  , 0  , 0  , 255, 255, 255, 255, 255, 255, 255, 255],
#     [0  , 0  , 0  , 0  , 255, 255, 255, 255, 255, 255, 255],
#     [0  , 0  , 0  , 0  , 0  , 255, 255, 255, 255, 255, 255],
#     [0  , 0  , 0  , 0  , 0  , 255, 255, 255, 255, 255, 255],
#     [0  , 0  , 0  , 0  , 0  , 0  , 255, 255, 255, 255, 255],
#     [0  , 255, 0  , 0  , 0  , 0  , 255, 255, 255, 255, 255],
#     [0  , 255, 255, 0  , 0  , 0  , 0  , 255, 255, 255, 255],
#     [0  , 255, 255, 255, 255, 0  , 0  , 0  , 255, 255, 255],
#     [0  , 255, 255, 255, 255, 255, 0  , 0  , 0  , 255, 255],
#     [0  , 255, 255, 255, 255, 255, 255, 0  , 0  , 0  , 255],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 0  , 0  , 255],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 0  , 0  ],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 0  , 0  ],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 0  , 0  ],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255, 0  ],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255, 0  ],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255, 0  ],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255, 0  ],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255, 0  ],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 0  , 0  ],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 0  , 0  ],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 0  , 255],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 0  , 0  , 255],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 0  , 255, 255],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
#     [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
# ], dtype = np.uint8) # Ratio W/H = 1/3

hookTemplate = np.array([
    [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255],
    [0  , 255, 255, 255, 255, 255, 255, 255, 255, 255],
    [0  , 0  , 255, 255, 255, 255, 255, 255, 255, 255],
    [0  , 0  , 255, 255, 255, 255, 255, 255, 255, 255],
    [0  , 0  , 0  , 255, 255, 255, 255, 255, 255, 255],
    [0  , 0  , 0  , 0  , 255, 255, 255, 255, 255, 255],
    [0  , 0  , 0  , 0  , 255, 255, 255, 255, 255, 255],
    [0  , 0  , 0  , 0  , 0  , 255, 255, 255, 255, 255],
    [255, 0  , 0  , 0  , 0  , 255, 255, 255, 255, 255],
    [255, 255, 0  , 0  , 0  , 0  , 255, 255, 255, 255],
    [255, 255, 255, 255, 0  , 0  , 0  , 255, 255, 255],
    [255, 255, 255, 255, 255, 0  , 0  , 0  , 255, 255],
    [255, 255, 255, 255, 255, 255, 0  , 0  , 0  , 255],
    [255, 255, 255, 255, 255, 255, 255, 0  , 0  , 255],
    [255, 255, 255, 255, 255, 255, 255, 255, 0  , 0  ],
    [255, 255, 255, 255, 255, 255, 255, 255, 0  , 0  ],
    [255, 255, 255, 255, 255, 255, 255, 255, 0  , 0  ],
    [255, 255, 255, 255, 255, 255, 255, 255, 255, 0  ],
    [255, 255, 255, 255, 255, 255, 255, 255, 255, 0  ],
    [255, 255, 255, 255, 255, 255, 255, 255, 255, 0  ],
    [255, 255, 255, 255, 255, 255, 255, 255, 255, 0  ],
    [255, 255, 255, 255, 255, 255, 255, 255, 255, 0  ],
    [255, 255, 255, 255, 255, 255, 255, 255, 0  , 0  ],
    [255, 255, 255, 255, 255, 255, 255, 255, 0  , 0  ],
    [255, 255, 255, 255, 255, 255, 255, 255, 0  , 255],
    [255, 255, 255, 255, 255, 255, 255, 0  , 0  , 255],
    [255, 255, 255, 255, 255, 255, 255, 0  , 255, 255],
], dtype = np.uint8) # Ratio W/H = 10/27

hookRotatedTemplate =  cv2.flip(hookTemplate, 0) # Ratio W/H = 1/3 or 10/27

# wholeRestTemplate = cv2.imread(r"C:\Users\nguye\Downloads\templates\rest\whole_rest.jpg", cv2.IMREAD_GRAYSCALE) # Ratio W/H = 58/85
wholeRestTemplate = cv2.imread(r"C:\Users\nguye\Downloads\templates\rest\whole_rest123.png", cv2.IMREAD_GRAYSCALE) # Ratio W/H = 24/39
# halfRestTemplate = cv2.imread(r"C:\Users\nguye\Downloads\templates\rest\half_rest_1.jpg", cv2.IMREAD_GRAYSCALE) # Ratio W/H = 58/85
halfRestTemplate = cv2.imread(r"C:\Users\nguye\Downloads\templates\rest\half_rest123.png", cv2.IMREAD_GRAYSCALE) # Ratio W/H = 24/39
quarterRestTemplate = cv2.imread(r"C:\Users\nguye\Downloads\templates\rest\quarter_rest.jpg", cv2.IMREAD_GRAYSCALE) # Ratio W/H = 19/74
eighthRestTemplate = cv2.imread(r"C:\Users\nguye\Downloads\templates\rest\eighth_rest.jpg", cv2.IMREAD_GRAYSCALE) # Ratio W/H = 20/74

W_H_RATIO_WHOLE_REST = 24/39
W_H_RATIO_HALF_REST = 24/39
W_H_RATIO_QUARTER_REST = 19/74
W_H_RATIO_EIGHTH_REST = 20/74