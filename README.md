# sheet_music_scanner

- Assignment 3: Sheet music scanner

- To-do:
1. Image pre-processing
2. Recognize the clefs to identify each line in the musical sheet; within the scope of this project, the program aims to identify the G clef and the F clef that mainly appears in the piano music sheet by using template matching
3. Line detection to identify the staff and the bar line by using the Hough transform algorithm
4. Notes detection (including the notes and the note lengths) by using the Hough transform algorithm to identify the outlines or template matching to fit the note at each frame and compute its position corresponding to the staff
5. Other elements detection: rests,...
6. Compute the frequency of each note and output the piano sound accordingly

- Progress:

1. Image processing: Not needed (already tried thresholding, blur + thresholding but it does not make the result significantly better)
2. Clef detection:
  - G Clef: Quite good
  - F Clef: Have not tried
3. Line detection: Using for loops through the whole music sheet (row by row, column by column):
  - Staff lines: works OK (can get coordinates), need to tune for 'threshold' to check for very extreme cases
  - Bar lines: specifically, the bar lines can be recognized (coordinates), but the veritcal lines in general is not
  - Another important information in this part can be extracted is 'staff lines gap': Done (measured in pixels)
  - Group 5 staff lines together to create a full line: Done (assume staff lines can be correctly detected)
4. Notes dectection: Using template matching for the note pitch:
  - Black, white, white alone notes: can be matched (coordinates) but have to tune 'threshold' for several different cases
  - Time (Duration): No progress has been made
5. Other elements:
  - Rests: template matching works quite well (have coordinates) for most of the cases, need to be aware of template size
6. And so on: Have not thought about.
