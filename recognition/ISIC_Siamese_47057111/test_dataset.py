#!/usr/bin/env python3
"""
Test script for ISIC dataset loader
Quick validation of dataset.py functionality
"""

import sys
import os

# Add the project directory to path
sys.path.append('/Users/matildadamman/COMP3710_A3/recognition/ISIC_Siamese_47057111')

# Change to project directory for relative paths
os.chdir('/Users/matildadamman/COMP3710_A3/recognition/ISIC_Siamese_47057111')

if __name__ == "__main__":
    from dataset import get_data_loaders, ISICDataset, ISICSiameseDataset
    
    print("Testing ISIC 2020 Dataset Loader...")
    
    # Run the main test
    exec(open('dataset.py').read())