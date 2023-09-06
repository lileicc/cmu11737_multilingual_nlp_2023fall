# Read in a text file and convert it to csv with 2 columns; "row_id" and "translation"
# This is to make your prediction file compatible with kaggle submission
import pandas as pd
import argparse

# Function to read text file and convert to CSV
def text_to_csv(text_filename, csv_filename):
    with open(text_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Remove newline characters from each line
    lines = [line.strip() for line in lines]

    # Create DataFrame
    df = pd.DataFrame({
        'row_id': range(1, len(lines) + 1),
        'translation': lines
    })

    # Save to CSV
    df.to_csv(csv_filename, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a text file to a CSV file.')
    parser.add_argument('input_file', type=str, help='The input text file.')
    parser.add_argument('output_file', type=str, help='The output CSV file.')
    
    args = parser.parse_args()

    text_to_csv(args.input_file, args.output_file)
