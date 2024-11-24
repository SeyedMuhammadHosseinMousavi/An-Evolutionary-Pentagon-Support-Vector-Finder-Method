# An evolutionary Pentagon Support Vector (PSV) finder method

- ### Please Cite:
- Mousavi, Seyed Muhammad Hossein, Vincent Charles, and Tatiana Gherman. "An Evolutionary Pentagon Support Vector Finder Method." Expert Systems with Applications 150 (2020): 113284.
- ### Link to the paper:
- https://www.sciencedirect.com/science/article/pii/S0957417420301093
  
<div align="justify">
This method is designed to improve classification tasks by reducing data size, removing outliers, 
and identifying support vectors using evolutionary algorithms and geometric computations. 
Below is a detailed explanation of the method's steps.
</div>

## Methodology:
![image](https://github.com/user-attachments/assets/5110a506-8f61-4803-9cff-1253e740e1cb)

1. **Load the Dataset:**
   - Begin with a dataset that contains features and labels.
   - Split the dataset into training and testing subsets to evaluate the method on unseen data.

2. **Evolutionary Clustering (ABC + FCM):**
   - Use Artificial Bee Colony (ABC) for optimization. Bees simulate data points in the clustering process.
   - Fuzzy C Means (FCM) is used for soft clustering, assigning probabilities to data points belonging to clusters.
   - Replace Euclidean distance in FCM with Manhattan distance to improve clustering performance.
![image](https://github.com/user-attachments/assets/ba83bea8-bc89-442b-b11c-d685e0fa566f)
![image](https://github.com/user-attachments/assets/e3b33479-5c04-40fc-a87c-18ae460a7ed8)

3. **Label Clusters with K-Nearest Neighbors (K-NN):**
   - After clustering, the data points in each cluster need to be labeled for classification.
   - Use K-NN to assign labels to clusters based on the proximity of their centers to the original training data.

4. **Outlier Removal Using Pentagon Area and Angles:**
   - Identify outliers by constructing a pentagon:
     - Select one sample from the current class and four samples from other classes.
     - Compute the area of the pentagon using the coordinates of its vertices.
     - Calculate the internal angles of the pentagon.
   - Apply thresholds:
     - If the pentagon's area exceeds a threshold, the sample is considered an outlier.
     - If any angle of the pentagon is outside the allowed range, the sample is also considered an outlier.
![image](https://github.com/user-attachments/assets/7c0f8b79-e613-4785-97a2-918eff180a30)

5. **Final Classification:**
   - Use Support Vector Machine (SVM) for classification.
   - Train the SVM on the reduced dataset (after clustering and outlier removal).
   - Compare the classification performance on:
     - The original dataset.
     - The reduced dataset (processed by the PSV method).

6. **Validation:**
   - Perform classification on benchmark datasets like Iris, Wine, and EEG Eye State.
   - Compare metrics such as accuracy, precision, recall, and runtime.
   - Analyze improvements in classification speed and accuracy.

## Key Advantages:
- Reduces computational load by removing unnecessary data points (outliers).
- Retains classification accuracy or improves it on certain datasets.
- Incorporates geometrical and evolutionary computations for robust data processing.

![image](https://github.com/user-attachments/assets/cd108804-bd0b-4e34-9803-e07c89afcadf)

![Evolutionary PSV](https://github.com/user-attachments/assets/32f5bcca-2a1a-4c9f-bac9-9bf0919739bd)

![image](https://github.com/user-attachments/assets/8a8eaeb9-c073-4b4a-981d-c8abe99c54b9)

- ### DOI:
- [https://www.sciencedirect.com/science/article/pii/S0957417420301093](https://doi.org/10.1016/j.eswa.2020.113284)
  
