# 🍕 PyTorch FoodVision: From Notebook to Web App  
### 🍕 FoodVision Mini (3-Class Classifier)  
- **Live Demo**: [foodvision_mini on Hugging Face Spaces](https://huggingface.co/spaces/Pranshulx26/foodvision_mini)  
- **Functionality**: Classifies images into "pizza", "steak", or "sushi".  
### 🍱 FoodVision Big (101-Class Classifier)  
- **Live Demo**: [FoodVision on Hugging Face Spaces](https://huggingface.co/spaces/Pranshulx26/FoodVision)  
- **Functionality**: Classifies images into one of 101 food categories from the Food101 dataset. Achieved ~58% accuracy on a 20% subset of the test data, surpassing the original Food101 paper's baseline of 56.4% accuracy.  
## 🚀 Project Overview  
This project demonstrates an end-to-end machine learning workflow, focusing on the deployment of PyTorch computer vision models as interactive web applications. Building upon the "FoodVision Mini" concept, this repository showcases the transition of a trained deep learning model from a Jupyter Notebook environment to a publicly accessible online demo using Gradio and Hugging Face Spaces. The project meticulously compares model performance, inference speed, and size to select the optimal model for real-world application.  
## ✨ Key Features  
- **End-to-End ML Workflow**: Covers data acquisition, model training, evaluation, and deployment.  
- **Transfer Learning**: Utilizes pre-trained EfficientNetB2 and Vision Transformer (ViT) feature extractors for efficient learning on custom datasets.  
- **Model Comparison**: Detailed analysis of model performance (accuracy, loss), inference speed (latency), and model size.  
- **Interactive Web Demos**: Deploys models as user-friendly web applications using Gradio.  
- **Cloud Deployment**: Hosts the Gradio demos on Hugging Face Spaces for public accessibility.  
- **Scalability Demonstration**: Extends from a 3-class "FoodVision Mini" to a 101-class "FoodVision Big" classifier.  
- **MLOps Principles**: Integrates concepts of model versioning, dependency management, and reproducible deployment.  
## 💡 Motivation  
The primary motivation behind this project is to bridge the gap between theoretical machine learning model development and practical, real-world application. A model's true value is realized when it can be used by others. This project aims to:  
- Demonstrate proficiency in deploying PyTorch models.  
- Showcase the importance of considering deployment constraints (e.g., speed, model size) alongside accuracy.  
- Provide a tangible, interactive example of a computer vision application.  
- Explore the capabilities of tools like Gradio and Hugging Face Spaces for rapid prototyping and sharing of ML demos.  
## 🛠️ Technical Stack  
- **Programming Language**: Python  
- **Deep Learning Framework**: PyTorch  
- **Computer Vision**: torchvision  
- **Model Summaries**: torchinfo  
- **Data Manipulation**: pandas, pathlib  
- **Interactive Demos**: Gradio  
- **Cloud Hosting**: Hugging Face Spaces  
- **Version Control**: Git (with Git LFS for large files)  
## 🧠 Model Architecture & Training  
Two state-of-the-art convolutional neural networks were experimented with as feature extractors:  
- **EfficientNetB2 (EffNetB2)**: A highly efficient model known for its balance of accuracy and computational efficiency.  
- **Vision Transformer (ViT-B/16)**: A transformer-based model that applies the transformer architecture directly to image patches.  
Both models were fine-tuned using transfer learning, where the pre-trained base layers were frozen, and only the classification head was trained on the custom dataset.  
## 📚 Dataset  
The project utilizes a subset of the Food101 dataset, which comprises 101 food categories with 1000 images each.  
- **FoodVision Mini**: Trained on a 20% subset of 3 classes (pizza, steak, sushi).  
- **FoodVision Big**: Trained on a 20% subset of all 101 classes. Data augmentation (TrivialAugmentWide) and label smoothing were applied during training for improved generalization.  
## 📊 Performance Comparison  
| Model                     | Test Loss | Test Accuracy (%) | Parameters     | Model Size (MB) | Time per Prediction (CPU, sec) |  
|---------------------------|-----------|-------------------|----------------|------------------|-------------------------------|  
| EffNetB2 Feature Extractor| 0.281     | 96.88             | 7,705,221      | 29               | 0.0269                        |  
| ViT-B/16 Feature Extractor| 0.064     | 98.47             | 85,800,963     | 327              | 0.0641                        |  
**Key Takeaway**: While ViT achieved slightly higher accuracy, EffNetB2 was chosen for deployment due to its significantly smaller model size and faster inference time, making it more suitable for mobile and web-based applications where speed is critical.  
## 🌐 Deployment  
The trained models were deployed as interactive web applications using **Gradio**, a Python library for quickly creating customizable UI components for ML models. These Gradio applications were then hosted on **Hugging Face Spaces**, a platform for sharing and showcasing machine learning demos.  
### 🍕 FoodVision Mini (3-Class Classifier)  
- **Live Demo**: [foodvision_mini on Hugging Face Spaces](https://huggingface.co/spaces/Pranshulx26/foodvision_mini)  
- **Functionality**: Classifies images into "pizza", "steak", or "sushi".  
### 🍱 FoodVision Big (101-Class Classifier)  
- **Live Demo**: [FoodVision on Hugging Face Spaces](https://huggingface.co/spaces/Pranshulx26/FoodVision)  
- **Functionality**: Classifies images into one of 101 food categories from the Food101 dataset. Achieved ~58% accuracy on a 20% subset of the test data, surpassing the original Food101 paper's baseline of 56.4% accuracy.  
## ⚙️ How to Run Locally  
To run this project on your local machine, follow these steps:  
**1. Clone the repository**  
```bash  
git clone https://github.com/Pranshulx26/Food_Classifier
cd FoodClassifier
```
**2. Navigate to the relevant directory**
```
cd demos/foodvision_mini   # or cd demos/FoodVision  
```
**3. Create and activate virtual environment**
```
python3 -m venv env  
source env/bin/activate    # On Windows: .\env\Scripts\activate  
```
**4. Install Dependencies**
```
pip install -r requirements.txt  
```
**5. Run the Gradio app```
```
python3 app.py  
```
## 🚀 Future Enhancements  
- Train on the **full Food101 dataset**  
- Add **GPU inference benchmarking**  
- Implement a **“Not Food” classifier** to filter irrelevant images  
- Explore **mobile/edge deployment** using PyTorch Mobile or TensorFlow Lite  
- Create a **FastAPI or TorchServe API endpoint**  
- Implement **CI/CD** for automated testing and deployment  
- Add a **user feedback loop** for continuous model improvement  

## 📧 Contact  
Feel free to connect with me for any questions or collaborations:

- **GitHub**: [https://github.com/Pranshulx26/](https://github.com/Pranshulx26/)  
- **Email**: pranshulsharma83@gmail.com  
