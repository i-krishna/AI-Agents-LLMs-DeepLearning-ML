# Architecture Diagram

<img width="152" height="584" alt="Architectuer" src="https://github.com/user-attachments/assets/a880ad47-5f24-4762-a9e8-1265639c6b5a" />

Model Building Steps 

1. Progressive Feature Extraction:
    * Early layers detect simple features (edges, colors)
    * Later layers detect complex patterns (shapes, objects)
2. Dimensionality Reduction:
    * Pooling reduces spatial dimensions while preserving important features
    * From 32x32 → 4x4 through the network
3. Parameter Efficiency:
    * Convolutional layers share parameters across spatial positions
    * Fewer parameters than fully connected networks for images
4. Hierarchical Learning:
    * Creates feature hierarchy from low to high level
