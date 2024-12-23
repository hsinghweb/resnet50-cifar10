class Config:
    # Dataset parameters
    DATASET_ROOT = './data'
    NUM_CLASSES = 10
    INPUT_SIZE = 32
    
    # Training parameters
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    # Model parameters
    PRETRAINED = True
    
    # Device configuration
    DEVICE = 'cuda'  # or 'cpu'
    
    # Logging and checkpoints
    CHECKPOINT_DIR = './checkpoints'
    LOG_INTERVAL = 100