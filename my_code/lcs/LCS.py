#!/usr/bin/python
# Authored by: Victoria O'Neill and Matthew Farr
# Modified by Juan Rojas
DB_PRINT=0

######################################################################
# Determine the score of the longest common subsequence of two strings
# students: complete this function
######################################################################
def longestCommonSubsequence(s1, s2):

     # set table size
     NUM_ROWS = len(s2)+1
     NUM_COLS = len(s1)+1
     lcs = ""

     # Create table and fill it with zeros
     costs = createTable(NUM_ROWS, NUM_COLS, 0)

     # Create table for getting back the optimal alignment, fill table with "A"
     directions = createTable(NUM_ROWS, NUM_COLS, "A")

     # Fill the top row of the costs table with zeroes and fill the top
     # row of the direction table with F's
     for i in range (0,NUM_COLS):
          costs[0][i] = 0
          directions[0][i] = "F"

     # Fill the left column of the costs table with zeroes and fill the
     # left column of the direction table with F's
     for j in range (1, NUM_ROWS):
          costs[j][0] = 0
          directions[j][0] = "F"

     maxRow = 0
     maxCol = 0
     maxVal = 0

     # Fill in each of the rows with subsequence scores, starting at the
     # top row and going right, then going down the rows
     for y in range (1,NUM_ROWS):
          for x in range (1,NUM_COLS):
               # Calculate costs of a gap for the top and bottom sequences
               valTop = costs[y-1][x]
               valLeft = costs[y][x-1]

               # Increase score for a match, or decrease for a mismatch
               #if (s1[x-1] == s2[y-1]):
               #     valDag = costs[y-1][x-1] + MATCH
               #else: valDag = costs[y-1][x-1] + MISMATCH

               # Calculate the maximum value and set that one as our cost,
               # then put the direction of the maximum value in the directions
               # table
               #val = max(valTop,valLeft,valDag)
               #costs[y][x] = val

               # Increase score for a match, otherwise ignore diagonal value
               if (s1[x-1] == s2[y-1]):
                    valDag = costs[y-1][x-1] + 1
               else:
                    valDag = -1 # Arbitrarily negative value so diagonal
                                # values aren't used unless there's a match

               # Set costs entry to highest of the three calculated values
               val = max(valTop, valLeft, valDag)
               costs[y][x] = val

               # If this is the highest value in the table, update the
               # highest value and its location
               if (val > maxVal):
                    maxVal = val
                    maxRow = y
                    maxCol = x

               # Set the direction table index to where the direction
               # the value came from
               if (val == valLeft):
                    directions[y][x] = "L"
               elif (val == valDag):
                    directions[y][x] = "D"
               else:
                    directions[y][x] = "T"
     
     # Print out table (only useful for small tables - used for debugging)
     # Comment out when you are satisfied that the algorithm is working
     #printTable(costs, "costs.txt")
     #printTable(directions, "directions.txt")

     # find optimal alignment
     lcs=align(directions, s1, s2, maxRow, maxCol, "alignment.txt")

     # return optimal score (lower right-hand cell in table)
     #return costs[maxRow][maxCol]
     return lcs

####################################################################
# Create a 2D table with the given number of rows and columns
# and fills all entries with value given as a parameter
# last arguments returns lcs
####################################################################
def createTable(numRows, numCols, value):
     table = []
     row = 0
     # create 2D table initialized with value
     while (row < numRows):
          table.append([])    
          col = 0
          while (col < numCols):
               table[row].append(value)
               col = col + 1
          row = row + 1
     return table

##################################################################
# Print 2D table to file (only useful for small tables for short
# strings)
# Should have tabs between the values on each row
# Useful function for debugging purposes
# Complete this function
##################################################################
def printTable(table, filename):

     file = open(filename, 'w')

     # Get the size of the table
     row = len(table)
     col = len(table[0])

     # Print each value in the table, separating columns with tab
     # characters and separating rows with newline characters
     for y in range (0,row):
          for x in range (0,col):
               file.write(str(table[y][x]) + "\t")
          file.write("\n")
     file.close()

     return
	

################################################################
# Reconstruct the optimal alignment and print the alignment
# to a file. Because the sequences can be long, print the 
# alignment 50 characters on one line, the other string of 50 characters
# on the next line, and then skip one line, as follows:
# AATT--GGCTATGCT--C-G-TTACGCA-TTACT-AA-TCCGGTC-AGGC
# AAATATGG---TGCTGGCTGCTT---CAGTTA-TGAACTCC---CCAGGC
#
# TATGGGTGCTATGCTCG--T--TACG-CA
# TCAT--TGG---TGCTGGCTGCTT--ACA
#
# Complete this function
# direction is a 2D table, seq1 and seq2 are the original DNA
# sequences to find the subsequence of and filename is the name
# of the output file
###############################################################
def align(direction, s1, s2, row, col, filename):
    lcs=''    
    if DB_PRINT:
        file = open(filename, 'w')
        
     # Set x and y coordinates for traversing the direction table
     # at the end of the highest common subsequence, and initializes
     # the subsequence strings as empty
    x = col
    y = row
    currentDir = direction[y][x]
             

    # Traverse the table until the top left corner of the table
    # is encountered
    while currentDir != "F":

         # If the direction index points up, move up the table,
         # and don't add anything to the string
         if direction[y][x] == "T":
              y -= 1
         # If the direction index points up, move left in the table,
         # and don't add anything to the string
         elif direction[y][x] == "L":
              x -= 1
         # If the direction index points diagonally, move up and left
         # in the table, and add a character from the strings to the
         # lcs
         elif direction[y][x] == "D":
              lcs = s1[x-1] + lcs               
              x-=1
              y-=1
         # Get next direction index from current position in the table
         currentDir = direction[y][x]

    # Print the longest common subsequence, 50 chars per line
    if DB_PRINT:
        for i in range (0, len(lcs), 50):
            file.write(lcs[i:i+50] + "\n")
        file.close()
    return lcs
### End of functions ###################################
