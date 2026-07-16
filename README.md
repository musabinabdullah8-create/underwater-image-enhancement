# underwater-image-enhancement
AI-powered underwater image enhancement and marine object detection using OpenCV, YOLOv8 and Streamlit.
Underwater Image Enhancement & Marine Object Detection Using Classical and Fusion-Based Algorithms

## 📌 Overview

This project is an AI-powered computer vision application developed to enhance underwater images and detect marine objects using deep learning techniques. Underwater images often suffer from low visibility, color distortion, blur, and poor contrast due to light absorption and scattering. This application enhances image quality before performing object detection, making underwater visual analysis more accurate.

The system integrates image enhancement techniques with the YOLOv8 object detection model and provides an interactive user interface using Streamlit.

## 🎯 Objectives

- Enhance underwater image quality.
- Improve visibility and color correction.
- Detect marine objects using YOLOv8.
- Provide an easy-to-use Streamlit interface.
- Evaluate detection performance on enhanced images.

## 🚀 Features

- Image upload through Streamlit
- Underwater image enhancement
- Contrast improvement
- Noise reduction
- Color correction
- YOLOv8-based marine object detection
- Bounding box visualization
- Detection confidence scores
- Performance evaluation
- Interactive graphical interface

## 🛠 Technologies Used

- Python
- Streamlit
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy
- Pandas
- Matplotlib


## 📂 Project Structure

```
underwater-image-enhancement/
│
├── app.py
├── enhancement.py
├── detection.py
├── requirements.txt
├── README.md
│
├── models/
│   └── yolov8.pt
│
├── images/
│   ├── input/
│   └── output/
│
├── screenshots/
│
├── report/
│   └── Underwater_Image_Enhancement_Report.pdf
│
└── presentation/
    └── Project_Presentation.pdf
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/underwater-image-enhancement.git
```

Move into the project directory

```bash
cd underwater-image-enhancement
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

```powershell
venv\Scripts\activate
```

After activation, you should see:

```powershell
(venv)
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run the following command inside the project folder:

```powershell
venv\Scripts\activate
```

You should immediately see:

```powershell
(venv)
```

at the beginning of the terminal.

Then launch the Streamlit application:

```powershell
streamlit run app.py
```

The application will automatically open in your default web browser.

---

## 📊 Workflow

1. Upload an underwater image.
2. Apply image enhancement techniques.
3. Improve visibility and contrast.
4. Perform marine object detection using YOLOv8.
5. Display detected objects with confidence scores.
6. Visualize enhanced and detected outputs.

---

## 📸 Results

The application successfully:

- Enhanced underwater image visibility.
- Improved contrast and color quality.
- Detected marine organisms using YOLOv8.
- Displayed bounding boxes with confidence scores.
- Produced visual outputs through Streamlit.

---

## 💡 Future Improvements

- Real-time underwater video enhancement.
- Support for multiple object detection models.
- Additional enhancement algorithms.
- GPU acceleration for faster inference.
- Cloud deployment.
- Integration with underwater robotics.

---

## 🎓 Academic Information

**Course:** Artificial Intelligence / Computer Vision Project

**Institution:** Air University, Multan Campus

---

## 👨‍💻 Author

**Musa Bin Abdullah**

Bachelor of Science in Computer Science

Air University, Multan Campus

LinkedIn: https://www.linkedin.com/in/musa-bin-abdullah-9277ab2b5

GitHub: https://github.com/musabinabdullah8

---

## 📄 License

This project is developed for educational and academic purposes.

MIT License.
