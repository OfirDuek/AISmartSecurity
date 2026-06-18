# AI Smart Security - Video Violence Detection

AI Smart Security is a machine-learning project for automatic violence detection in short video clips.

The system receives a video as input, performs preprocessing, runs a trained classification model, and returns a binary result indicating whether violence was detected in the video.

The project focuses on the violence detection model and its inference pipeline. It does not include a full user interface, database, user management system, or real-time camera integration.

---

## Project Goal

The goal of this project is to develop a machine-learning model capable of classifying video clips into one of two classes:

- **Violence** - the video contains violent behavior.
- **Non-Violence / Routine** - the video does not contain violent behavior.

The system is intended to serve as a technological basis for future smart security systems that can assist in identifying suspicious or violent events from video footage.

---

## Main Features

- Accepts video input as a local video file or an online video link.
- Performs preprocessing on the input video.
- Splits the video into short clips of approximately 3 seconds.
- Extracts 16 frames from each clip.
- Resizes and normalizes frames before model inference.
- Loads a trained video classification model.
- Classifies each clip as violent or non-violent.
- Produces a final binary decision for the full video.

---

## System Output

The system returns a clear binary result:

Violence: Yes

or

Violence: No

If at least one clip in the video is classified as violent, the full video is classified as containing violence.

---

## Model

The final model used in this project is:

**MViT V2 S - Multiscale Vision Transformer**

This model was selected because it is suitable for video-based classification tasks and can analyze visual patterns across a sequence of frames.

The model receives a processed video clip represented by 16 frames and predicts whether the clip contains violent content.

---

## Baseline Model

As part of the development process, an initial baseline model was tested:

**ResNet50 + LSTM**

The baseline model was used as a reference point for evaluating the performance of the final model.

Although it achieved high recall, it produced many false positive classifications and did not separate the two classes well enough.

Baseline results:

| Metric | Result |
|---|---:|
| Train Accuracy | 0.4963 |
| Train Loss | 0.6960 |
| Validation Accuracy | 0.5260 |
| Validation Loss | 0.6921 |
| Precision | 0.5134 |
| Recall | 0.9809 |
| F1 Score | 0.6740 |

Following these results, the project moved to the MViT V2 S model, which achieved significantly better classification performance.

---

## Final Model Results

The final model achieved strong performance on the validation/evaluation data:

| Metric | Result |
|---|---:|
| Validation Accuracy | 96% - 97% |
| F1 Score | ~0.96 |
| AUC | 0.9756 |
| Correct Classifications | 443 / 471 |

The results indicate that the final model provides a strong ability to distinguish between violent and non-violent video clips.

---

## Project Structure

AISmartSecurity/
│
├── trained_model/
│   └── trained model files
│
├── videos/
│   └── input videos
│
├── model.py
├── preprocess.py
├── utils.py
├── requirements.txt
└── README.md

### Main Files

- `model.py` - loads the trained model and performs prediction.
- `preprocess.py` - handles video preprocessing, frame extraction, resizing, and normalization.
- `utils.py` - contains helper functions used by the system.
- `requirements.txt` - lists the required Python dependencies.
- `trained_model/` - contains the trained model file.
- `videos/` - contains input videos used for testing or inference.

---

## Preprocessing Pipeline

Before running inference, each video goes through the following preprocessing steps:

1. The video is loaded from a local file or downloaded from an online link.
2. The video is divided into short clips of approximately 3 seconds.
3. 16 frames are extracted from each clip.
4. Each frame is resized to `224x224`.
5. Frame values are normalized according to the preprocessing used during training.
6. The processed frames are arranged into the input structure expected by the model.

---

## Technologies Used

The project is implemented in Python and uses the following main libraries:

- Python
- PyTorch
- TorchVision
- OpenCV
- NumPy
- yt-dlp
- Matplotlib
- scikit-learn

---

## Installation

Clone the repository:

git clone <repository-url>
cd AISmartSecurity

Install the required dependencies:

pip install -r requirements.txt

---

## Usage

Run the main inference script with a local video file or video link.

Example with a local video file:

python main.py --input videos/example.mp4

Example with an online video link:

python main.py --input "<video-url>"

Note: The exact run command may change depending on the final main file implementation.

---

## Limitations

The current project focuses on the model and inference pipeline only.

The following components are not included in the current implementation:

- Full user interface
- Web application
- Database
- User authentication or permissions
- Real-time camera stream integration
- Automatic alert system
- Event management or investigation system

The model is intended to support initial detection and should not be used as a full replacement for human judgment in sensitive or operational decisions.

---

## Future Work

Possible future improvements include:

- Integration with live camera streams.
- Development of a full user interface.
- Real-time alert generation.
- Expansion to multi-class violence classification.
- Localization of violent activity within the video frame.
- Further training on larger and more diverse datasets.

---

## Team Members

- Ofir Duek
- Michael Yehoshua
- Aviv Meir
- Rotem Aloni
- Yarin Guberman

Machine Learning Final Project, 2026.
