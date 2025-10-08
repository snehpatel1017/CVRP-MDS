#!/bin/bash

# --- Configuration ---
# IMPORTANT: Change this to the actual command to run your C++ program.
# For example, if your compiled program is named 'cvrp_solver', use "./cvrp_solver".
EXECUTABLE="./clareke_and_wright.out"

# --- Directory Setup ---
INPUT_DIR="inputs"
OUTPUT_DIR="Outputs"

# --- Script Logic ---

# Exit if the input directory doesn't exist
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory '$INPUT_DIR' not found."
  echo "Please make sure the 'Inputs' folder exists in the same directory as this script."
  exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting CVRP batch processing..."

# Loop through all files in the input directory
for input_file in "$INPUT_DIR"/*
do
  # Check if it's a file before processing
  if [ -f "$input_file" ]; then
    # Get the base name of the file (e.g., 'toy.vrp' from 'Inputs/toy.vrp')
    filename=$(basename "$input_file")

    # Define the path for the output file
    output_file="$OUTPUT_DIR/$filename"

    echo "Processing: $filename"

    # Run your executable, redirecting the input file to its standard input,
    # and redirecting its standard output to the new output file.
    $EXECUTABLE  "$input_file" > "$output_file"
  fi
done

echo "------------------------------------"
echo "Batch processing complete."
echo "All outputs have been saved in the '$OUTPUT_DIR' directory."
