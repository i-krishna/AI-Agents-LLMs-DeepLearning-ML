# create a convolutional neural network (CNN) for image classification.

# It follows a classic CNN pattern of:
# Convolutional layers for feature extraction
# Pooling layers for dimensionality reduction
# Dense layers for classification

model = models.Sequential([
    # Convolutional layers

    # First Convolutional Block

    # Conv2D: 2D convolutional layer
    # 32: Number of filters/kernels (creates 32 feature maps)
    # (3, 3): Kernel size (3x3 pixels)
    # activation='relu': ReLU activation (Rectified Linear Unit) introduces non-linearity
    # input_shape=(32, 32, 3): Input images are 32x32 pixels with 3 channels (RGB)
    
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    
    # First Max Pooling Layer

    # (2, 2): Pooling window size. Reduces spatial dimensions by taking maximum value in each 2x2 window
    # Output: (15, 15, 32) (halves the width and height)

    layers.MaxPooling2D((2, 2)),

    # Second Convolutional Block

    # 64 filters: More filters to capture higher-level features
    # Output: (13, 13, 64) (again loses border pixels)

    layers.Conv2D(64, (3, 3), activation='relu'),

    # Second Max Pooling Layer
    # Output: (6, 6, 64) (13/2 = 6.5 → rounds down to 6)

    layers.MaxPooling2D((2, 2)),

    # Third Convolutional Layer
    # No pooling after this one - allows more feature extraction before flattening
    # Output: (4, 4, 64) (6-3+1=4)

    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Dense layers
    
    # Converts 3D feature maps (4,4,64) to 1D vector (4*4*64=1024 dimensions). 
    # Prepares data for dense layers
    layers.Flatten(),

    # First Dense Layer
    # Fully connected layer with 64 neurons. ReLU activation for non-linearity
    # Learns combinations of features from convolutional layers
    layers.Dense(64, activation='relu'),

    # Output Layer
    # 10 neurons for 10-class classification
    # No activation (raw logits) because we'll use from_logits=True in loss function
    # For multi-class classification, softmax would typically be applied later
    layers.Dense(10)
])

# Compile the model

# Optimizer: Adam (adaptive learning rate optimization)
# Loss: Sparse Categorical Crossentropy (for integer labels, more memory efficient than one-hot)
# from_logits=True means the model outputs raw scores (no softmax)
# Metrics: Accuracy (fraction of correctly classified images)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Display model architecture
model.summary()