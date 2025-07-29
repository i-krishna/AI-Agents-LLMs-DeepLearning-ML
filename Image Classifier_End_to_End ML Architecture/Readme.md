# Architecture Diagram

<img width="81" height="295" alt="archi" src="https://github.com/user-attachments/assets/188fb3bc-d108-4e1f-9fcd-35f4521219bb" />

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
