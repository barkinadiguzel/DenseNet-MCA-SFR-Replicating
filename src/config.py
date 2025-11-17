# DenseNet settings
GROWTH_RATE = 32
BLOCK_LAYERS = [6, 12, 24, 16]  # Based on DenseNet-121
NUM_CLASSES = 1000

# MCA settings
MCA_OUT_CHANNELS = GROWTH_RATE * 2

# SFR settings
SFR_PROB = 0.8

# Input size
INPUT_CHANNELS = 3
INPUT_SIZE = (224, 224)
