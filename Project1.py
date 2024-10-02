'''
Name : Shreejan Pandey
Student Number : 800787174

Running on the server. 

The Structure of the folder having this script files should look like
    P1AASeqs
        :->sequenceA1.txt
        :->sequenceA2.txt
        :->sequenceB1.txt
        :->sequenceB2.txt
        :->sequenceC1.txt
        :->sequenceC2.txt
    P1SubMatrices
        :->AAnucleoPP.txt
        :->BLOSUM62.txt
        :->HP.txt
        :->PAM250-scores.txt
    Project1.py

Once the structure is there running  command = python Project1.py will run the code on the server.

'''

import numpy as np  # For numerical operations and handling matrices
import os  # For handling file paths

def read_sequence(file_path):
    """Reads a sequence file and filters out non-alphabetic characters, ignoring the first line."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) > 1:
            # Return the second line and filter non-alphabetic characters
            return ''.join(filter(str.isalpha, lines[1].strip()))
        else:
            return ''

def read_substitution_matrix(file_path):
    """Reads a substitution matrix from a file and returns it as a dictionary, ensuring symmetry."""
    substitution_matrix = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        headers = lines[1].strip().split(',')  # Assuming the second line contains column headers
        
        # Infer row characters based on the order of headers
        row_chars = headers  # Assuming rows are in the same order as columns     
        for row_index, line in enumerate(lines[2:]):
            parts = line.strip().split(',')
            row_char = row_chars[row_index]
            for col_index, value in enumerate(parts):
                score = float(value)
                col_char = headers[col_index]
                substitution_matrix[(row_char, col_char)] = score
                substitution_matrix[(col_char, row_char)] = score  # Ensure symmetry
    return substitution_matrix

def global_alignment(seq1, seq2, matrix, gap_penalty):
    """Performs Global Alignment using the Needleman-Wunsch algorithm."""
    n = len(seq1)
    m = len(seq2)
    dp = np.zeros((n+1, m+1), dtype=float)
    
    # Initialize the first column and first row of the DP matrix with gap penalties
    for i in range(n+1):
        dp[i][0] = i * gap_penalty
    for j in range(m+1):
        dp[0][j] = j * gap_penalty
    
    # Fill the DP matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            match = dp[i-1][j-1] + matrix.get((seq1[i-1], seq2[j-1]), float('-inf'))
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match, delete, insert)
            # Debug: Uncomment the following line to see the DP matrix being filled
            # print(f"dp[{i}][{j}] = max({match}, {delete}, {insert}) = {dp[i][j]}")
    
    # Backtracking to find the optimal alignment
    align1, align2 = '', ''
    i, j = n, m
    
    while i > 0 and j > 0:
        score = dp[i][j]
        score_diag = dp[i-1][j-1]
        score_up = dp[i][j-1]
        score_left = dp[i-1][j]
        
        # Check which operation was used to reach the current cell
        if score == score_diag + matrix.get((seq1[i-1], seq2[j-1]), float('-inf')):
            align1 += seq1[i-1]
            align2 += seq2[j-1]
            i -= 1
            j -= 1
        elif score == score_left + gap_penalty:
            align1 += seq1[i-1]
            align2 += '-'
            i -= 1
        elif score == score_up + gap_penalty:
            align1 += '-'
            align2 += seq2[j-1]
            j -= 1
    
    # Handle remaining gaps in the alignment
    while i > 0:
        align1 += seq1[i-1]
        align2 += '-'
        i -= 1
    while j > 0:
        align1 += '-'
        align2 += seq2[j-1]
        j -= 1
    
    # Reverse the alignments as we built them from the end
    return dp, align1[::-1], align2[::-1], dp[n][m]

def local_alignment(seq1, seq2, matrix, gap_penalty):
    """Performs Local Alignment using the Smith-Waterman algorithm."""
    n = len(seq1)
    m = len(seq2)
    dp = np.zeros((n+1, m+1), dtype=float)
    max_score = 0
    max_pos = (0, 0)
    
    # Fill the DP matrix with scores, allowing resets to zero for local alignment
    for i in range(1, n+1):
        for j in range(1, m+1):
            match = dp[i-1][j-1] + matrix.get((seq1[i-1], seq2[j-1]), float('-inf'))
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(0, match, delete, insert)  # Local alignment allows starting anew
            if dp[i][j] >= max_score:
                max_score = dp[i][j]
                max_pos = (i, j)
            # Debug: Uncomment the following line to see the DP matrix being filled
            # print(f"dp[{i}][{j}] = max(0, {match}, {delete}, {insert}) = {dp[i][j]}")
    
    # Backtracking to find the optimal local alignment
    align1, align2 = '', ''
    i, j = max_pos
    
    while dp[i][j] != 0:
        score = dp[i][j]
        score_diag = dp[i-1][j-1]
        score_up = dp[i][j-1]
        score_left = dp[i-1][j]
        
        # Determine the direction of the traceback
        if score == score_diag + matrix.get((seq1[i-1], seq2[j-1]), float('-inf')):
            align1 += seq1[i-1]
            align2 += seq2[j-1]
            i -= 1
            j -= 1
        elif score == score_left + gap_penalty:
            align1 += seq1[i-1]
            align2 += '-'
            i -= 1
        elif score == score_up + gap_penalty:
            align1 += '-'
            align2 += seq2[j-1]
            j -= 1
    
    # Reverse the alignments as we built them from the end
    return dp, align1[::-1], align2[::-1], max_score

def semi_global_alignment(seq1, seq2, matrix, gap_penalty):
    """Performs Semi-Global Alignment."""
    n = len(seq1)
    m = len(seq2)
    dp = np.zeros((n+1, m+1), dtype=float)
    
    # Initialize the first row and first column to zero for semi-global alignment
    for i in range(1, n+1):
        dp[i][0] = 0
    for j in range(1, m+1):
        dp[0][j] = 0
    
    # Fill the DP matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            match = dp[i-1][j-1] + matrix.get((seq1[i-1], seq2[j-1]), float('-inf'))
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match, delete, insert)
            # Debug: Uncomment the following line to see the DP matrix being filled
            # print(f"dp[{i}][{j}] = max({match}, {delete}, {insert}) = {dp[i][j]}")
    
    # Determine the end position for backtracking
    max_i = np.argmax(dp[:, m])
    max_j = np.argmax(dp[n, :])
    
    if dp[max_i, m] > dp[n, max_j]:
        align1, align2 = '', ''
        i, j = max_i, m
    else:
        align1, align2 = '', ''
        i, j = n, max_j
    
    # Backtracking to find the optimal semi-global alignment
    while i > 0 and j > 0:
        score = dp[i][j]
        score_diag = dp[i-1][j-1]
        score_up = dp[i][j-1]
        score_left = dp[i-1][j]
        
        if score == score_diag + matrix.get((seq1[i-1], seq2[j-1]), float('-inf')):
            align1 += seq1[i-1]
            align2 += seq2[j-1]
            i -= 1
            j -= 1
        elif score == score_left + gap_penalty:
            align1 += seq1[i-1]
            align2 += '-'
            i -= 1
        elif score == score_up + gap_penalty:
            align1 += '-'
            align2 += seq2[j-1]
            j -= 1
    
    # Handle remaining gaps in the alignment
    while i > 0:
        align1 += seq1[i-1]
        align2 += '-'
        i -= 1
    while j > 0:
        align1 += '-'
        align2 += seq2[j-1]
        j -= 1
    last_row = dp[n, :]
    last_col = dp[:, m]
    max_score_row = np.max(last_row)
    max_score_col = np.max(last_col)

    optimal_score = max(max_score_row, max_score_col)
    # Reverse the alignments as we built them from the end
    return dp, align1[::-1], align2[::-1], optimal_score

def global_alignment_affine(seq1, seq2, matrix, gap_open, gap_extend):
    """Performs Global Alignment using Affine Gap Penalty."""
    n = len(seq1)
    m = len(seq2)
    
    # Initialize three matrices: M for match/mismatch, Ix for gaps in seq1, Iy for gaps in seq2
    M = np.full((n+1, m+1), float('-inf'))
    Ix = np.full((n+1, m+1), float('-inf'))
    Iy = np.full((n+1, m+1), float('-inf'))
    
    # Initialize gap penalties for the first column and first row
    for i in range(1, n+1):
        Ix[i][0] = gap_open + (i-1) * gap_extend
    for j in range(1, m+1):
        Iy[0][j] = gap_open + (j-1) * gap_extend
    
    M[0][0] = 0  # Starting point
    
    # Fill the DP matrices
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Update M matrix: match or mismatch
            M[i][j] = max(
                M[i-1][j-1] + matrix.get((seq1[i-1], seq2[j-1]), float('-inf')),
                Ix[i-1][j-1] + matrix.get((seq1[i-1], seq2[j-1]), float('-inf')),
                Iy[i-1][j-1] + matrix.get((seq1[i-1], seq2[j-1]), float('-inf'))
            )
            
            # Update Ix matrix: gap in seq1
            Ix[i][j] = max(
                M[i-1][j] + gap_open + gap_extend,  # Opening a new gap
                Ix[i-1][j] + gap_extend  # Extending an existing gap
            )
            
            # Update Iy matrix: gap in seq2
            Iy[i][j] = max(
                M[i][j-1] + gap_open + gap_extend,  # Opening a new gap
                Iy[i][j-1] + gap_extend  # Extending an existing gap
            )
    
    # Backtracking to find the optimal alignment
    align1, align2 = '', ''
    i, j = n, m
    
    # Determine which matrix has the highest score at the bottom-right corner
    end_scores = {'M': M[n][m], 'Ix': Ix[n][m], 'Iy': Iy[n][m]}
    current_matrix = max(end_scores, key=end_scores.get)
    optimal_score = end_scores[current_matrix]
    while i > 0 and j > 0:
        if current_matrix == 'M':
            # Check from which matrix the current cell was derived
            if M[i][j] == M[i-1][j-1] + matrix.get((seq1[i-1], seq2[j-1]), float('-inf')):
                align1 += seq1[i-1]
                align2 += seq2[j-1]
                i -= 1
                j -= 1
                current_matrix = 'M'
            elif M[i][j] == Ix[i-1][j-1] + matrix.get((seq1[i-1], seq2[j-1]), float('-inf')):
                align1 += seq1[i-1]
                align2 += seq2[j-1]
                i -= 1
                j -= 1
                current_matrix = 'Ix'
            elif M[i][j] == Iy[i-1][j-1] + matrix.get((seq1[i-1], seq2[j-1]), float('-inf')):
                align1 += seq1[i-1]
                align2 += seq2[j-1]
                i -= 1
                j -= 1
                current_matrix = 'Iy'
        elif current_matrix == 'Ix':
            # Moving vertically in Ix matrix indicates a gap in seq2
            align1 += seq1[i-1]
            align2 += '-'
            i -= 1
            # Determine whether the gap was opened or extended
            if Ix[i][j] == M[i][j] + gap_open + gap_extend:
                current_matrix = 'M'
            else:
                current_matrix = 'Ix'
        elif current_matrix == 'Iy':
            # Moving horizontally in Iy matrix indicates a gap in seq1
            align1 += '-'
            align2 += seq2[j-1]
            j -= 1
            # Determine whether the gap was opened or extended
            if Iy[i][j] == M[i][j] + gap_open + gap_extend:
                current_matrix = 'M'
            else:
                current_matrix = 'Iy'
    
    # Handle any remaining gaps after reaching the top or left edge
    while i > 0:
        align1 += seq1[i-1]
        align2 += '-'
        i -= 1
    while j > 0:
        align1 += '-'
        align2 += seq2[j-1]
        j -= 1
    
    # Reverse the alignments as we built them from the end
    return M, Ix, Iy, align1[::-1], align2[::-1], optimal_score

def main():
    # Base path where sequence files are located
    base_path = './P1AASeqs' 

    # Prompt user to select the alignment algorithm
    print("Select alignment algorithm:")
    print("1. Global Alignment")
    print("2. Local Alignment")
    print("3. Semi-Global Alignment")
    print("4. Global Alignment with Affine Gap Penalty")
    
    try:
        algorithm_choice = int(input("Enter choice (1/2/3/4): "))
    except ValueError:
        print("Invalid input! Please enter a number between 1 and 4.")
        return
    
    # Prompt user to select the sequence set
    print("\nSelect string set:")
    print("A. Set A consist of files (SequenceA1.txt and SequenceA2.txt)")
    print("B. Set B consist of files (SequenceB1.txt and SequenceB2.txt)")
    print("C. Set C consist of files (SequenceC1.txt and SequenceC2.txt)")
    
    set_selection = input("Enter choice (A/B/C): ").upper()
    
    # Determine file paths based on user selection
    if set_selection == 'A':
        sequence_file_1 = os.path.join(base_path, 'sequenceA1.txt')
        sequence_file_2 = os.path.join(base_path, 'sequenceA2.txt')
    elif set_selection == 'B':
        sequence_file_1 = os.path.join(base_path, 'sequenceB1.txt')
        sequence_file_2 = os.path.join(base_path, 'sequenceB2.txt')
    elif set_selection == 'C':
        sequence_file_1 = os.path.join(base_path, 'sequenceC1.txt')
        sequence_file_2 = os.path.join(base_path, 'sequenceC2.txt')
    else:
        print("Invalid set selection!")
        return
    
    # Read the selected sequences from files
    try:
        sequence_1 = read_sequence(sequence_file_1)
        sequence_2 = read_sequence(sequence_file_2)
    except FileNotFoundError as e:
        print(f"Error reading sequence files: {e}")
        return
    
    # Prompt user to input the substitution matrix file name
    subtitution_matrix_folder_path = './P1SubMatrices/'
    substitution_matrix_path = input("\nEnter substitution matrix name (e.g., AAnucleoPP.txt, BLOSUM62.txt, HP.txt, PAM250-scores.txt): ").strip()
    try:
        substitution_matrix = read_substitution_matrix(os.path.join(subtitution_matrix_folder_path, substitution_matrix_path))
    except FileNotFoundError as e:
        print(f"Error reading substitution matrix file: {e}")
        return
    except ValueError as ve:
        print(f"Error parsing substitution matrix file: {ve}")
        return
    
    # Handle Global Alignment with Affine Gap Penalty separately
    if algorithm_choice == 4:
        try:
            gap_open_penalty = float(input("Enter gap open penalty (e.g., -5): "))
            gap_extend_penalty = float(input("Enter gap extend penalty (e.g., -1): "))
        except ValueError:
            print("Invalid input! Please enter numeric values for penalties.")
            return
        
        try:
            # Perform Global Alignment with Affine Gap Penalty
            M_matrix, Ix_matrix, Iy_matrix, aligned_seq_a, aligned_seq_b, optimal_Score = global_alignment_affine(
                sequence_1, sequence_2, substitution_matrix, gap_open_penalty, gap_extend_penalty
            )
            print("\nOptimal Alignment:")
            print(aligned_seq_a)
            print(aligned_seq_b)
            print("M matrix:\n", M_matrix)
            print("Ix matrix:\n", Ix_matrix)
            print("Iy matrix:\n", Iy_matrix)
            print("Optimal Score:", optimal_Score)
        except ValueError as ve:
            print(f"Error during alignment: {ve}")
            return
    else:
        try:
            # Prompt user to input the gap penalty
            gap_penalty = float(input("Enter gap penalty (e.g., -5): "))
        except ValueError:
            print("Invalid input! Please enter a numeric value for gap penalty.")
            return
        
        try:
            # Perform the selected alignment based on user's choice
            if algorithm_choice == 1:
                # Global Alignment
                dp_matrix, aligned_seq_a, aligned_seq_b, optimal_Score = global_alignment(
                    sequence_1, sequence_2, substitution_matrix, gap_penalty
                )
            elif algorithm_choice == 2:
                # Local Alignment
                dp_matrix, aligned_seq_a, aligned_seq_b, optimal_Score = local_alignment(
                    sequence_1, sequence_2, substitution_matrix, gap_penalty
                )
            elif algorithm_choice == 3:
                # Semi-Global Alignment
                dp_matrix, aligned_seq_a, aligned_seq_b, optimal_Score = semi_global_alignment(
                    sequence_1, sequence_2, substitution_matrix, gap_penalty
                )
            else:
                print("Invalid algorithm choice!")
                return
        except ValueError as ve:
            print(f"Error during alignment: {ve}")
            return
        
        # Display the optimal alignment and the DP matrix
        print("\nOptimal Alignment:")
        print(aligned_seq_a)
        print(aligned_seq_b)
        print("Score matrix:\n", dp_matrix)
        print("Optimal Score", optimal_Score)

if __name__ == "__main__":
    main()
