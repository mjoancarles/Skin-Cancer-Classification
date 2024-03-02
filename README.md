# Skin Cancer Classification Project 🚀

## Overview

This project, titled "Skin Cancer Classification," is developed as a final university project to explore and innovate in the field of medical imaging, specifically focusing on the classification of skin lesions. The aim is to provide a preliminary approach to identifying whether a small lesion could be melanoma, one of the most lethal cancers. Given the importance of early detection, this project utilizes various techniques, including a Convolutional Neural Network (CNN) built with TensorFlow, to classify seven different types of lesions.

### Importance

Skin cancer, particularly melanoma, poses a significant health risk due to its high fatality rate if not detected early. According to the [SEER Cancer Statistics](https://seer.cancer.gov/statfacts/html/melan.html), understanding and identifying melanoma early can significantly impact treatment outcomes. This project strives to contribute to the early detection efforts by providing a tool for classifying skin lesions.

### Data Source

The dataset used for this project is the HAM10000 dataset, which has been sourced from a [Kaggle repository](https://www.kaggle.com/datasets/volodymyrpivoshenko/skin-cancer-lesions-segmentation) containing masks of the lesions. This dataset is pivotal for training our models to recognize and differentiate between various types of skin lesions.

## Project Structure

- **Two-fold Approach:** The project employs a dual strategy for lesion classification:
  - **7-Class Classification:** Utilizes classical Machine Learning techniques locally and a CNN model in Google Colab for classifying the lesions into seven categories.
  - **Binary Classification:** Focuses on distinguishing melanomas from other lesion types, implemented both locally and in Google Colab.

- **Heatmaps Analysis:** To understand what features the model captures from each lesion, heatmaps have been generated, offering insights into the decision-making process of the CNN.

- **Streamlit Application:** For practical application, a Streamlit app has been developed, allowing users to test the model with images of lesions. This application is accessible via a Docker container:
  ```
  docker pull mjoancarles/skincancer:latest
  ```

### Additional Resources

For a comprehensive understanding of the project, the following resources are recommended:
- 📄 **Article.pdf:** A detailed article explaining the methodologies, findings, and implications of the project.
- 🎥 **PPT_Presentation.pdf:** A presentation providing an overview and key insights from the project.

## Disclaimer

As this project serves as an academic endeavor, it's important to note that the models and techniques employed may not be precise enough for clinical use and could contain errors. It should be seen as a first step towards leveraging AI in skin cancer detection, necessitating further validation and improvement.

## How to Contribute

Contributions to enhance the project, refine the models, or improve the application are welcome. Feel free to fork the repository, make your changes, and submit a pull request.

## Acknowledgments

Special thanks to the providers of the HAM10000 dataset and the Kaggle community for making the data accessible, enabling this project to take shape.