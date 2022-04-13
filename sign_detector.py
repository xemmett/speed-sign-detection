"""
Kacper Dudek - 18228798
Christian Ryan - 18257356
Charlie Gorey O'Neill - 18222803
Sean McTiernan - 18224385
"""

from region_proposer import RegionProposer
from classifier import classifier
from os import listdir


if __name__ == "__main__":
    names_list = []
    label_loc_list = []

    for filename in listdir('speed-sign-test-images'):
        if(filename.endswith('.png')):
            region = RegionProposer(f"speed-sign-test-images/{filename}", verbose=False)

    for filename in listdir('roi'):
        if(filename.endswith('.png')):
            classifier('roi', filename)