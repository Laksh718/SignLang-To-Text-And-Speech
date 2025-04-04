# Sign Language To Text and Speech Conversion

## Project Overview

This application converts American Sign Language (ASL) gestures into text and speech in real-time using a convolutional neural network (CNN). Developed by Laksh Baweja and team, it captures video through a webcam, recognizes hand gestures, converts them to text, and reads the text aloud using text-to-speech conversion.

## Demo

Click the image below to watch the demo:

[![ASL to Text and Speech Conversion Demo](https://img.youtube.com/vi/mnpAdZGfyAI/0.jpg)](https://youtu.be/mnpAdZGfyAI)

## Key Features

- **Real-time ASL gesture recognition** using a webcam
- **Conversion to text** with word suggestions
- **Text-to-speech functionality** with multiple voice options
- **Visual indicators** for hand detection status
- **Space and next gesture controls** for forming sentences
- **Clean, intuitive user interface** with team branding

## Recent Improvements

- Added visual indicators for hand detection status
- Improved space gesture handling with two-step confirmation (space + next)
- Enhanced next gesture cooldown to prevent accidental repetition
- Updated UI with team name display
- Improved error handling and debugging information
- Added multiple voice options with a toggle feature

## How to Use

1. Run the application: `python3 final_pred.py`
2. Position your hand in front of the camera
3. Make ASL gestures corresponding to the letters you want to spell
4. Use the space gesture followed by next gesture to add spaces
5. Use the next gesture to confirm characters
6. Use the buttons at the bottom to:
   - Speak the current text
   - Clear the text
   - Toggle through available voices

### Gesture Controls

- **Character Input**: Make the corresponding ASL hand gesture
- **Space**: Make the space gesture (similar to "I" but with pinky extended)
- **Next**: Confirm the current character/space and add it to the sentence
- **Backspace**: Use the backspace gesture to delete characters

## System Requirements

- **Operating System**: Windows/Mac/Linux
- **Python**: 3.6 or higher
- **Camera**: Webcam with clear view of hands
- **Libraries**:
  - OpenCV
  - TensorFlow/Keras
  - Mediapipe
  - NumPy
  - pyttsx3
  - Tkinter
  - PIL

## Technical Implementation

### Hand Detection

The application uses the MediaPipe library to detect hand landmarks in real-time. It processes these landmarks to identify specific hand poses corresponding to ASL gestures.

### Gesture Classification

A CNN model (cnn8grps_rad1_model.h5) trained on ASL gestures classifies the detected hand poses. The model divides the 26 alphabet gestures into 8 groups to improve accuracy, achieving 97-99% accuracy depending on lighting conditions. The groups are:

- Group 1: [y, j]
- Group 2: [c, o]
- Group 3: [g, h]
- Group 4: [b, d, f, i, u, v, k, r, w]
- Group 5: [p, q, z]
- Group 6: [a, e, m, n, s, t]
- Group 7 & 8: Additional classifications

### Training Approach

To overcome background and lighting issues, the team implemented a novel approach:

1. Detect hand landmarks using MediaPipe
2. Draw these landmarks on a plain white background
3. Use these skeleton images for training the CNN model
4. Use mathematical operations on hand landmarks to further classify gestures within each group

### Text Generation and Speech Conversion

Recognized gestures are converted to text and displayed on the screen. The application uses pyttsx3 to convert the text to speech when requested.

## Project Structure

- `final_pred.py`: Main application file with UI and gesture recognition logic
- `white.jpg`: Background image used for visualization
- `cnn8grps_rad1_model.h5`: Trained CNN model for gesture recognition
- `AtoZ_3.1/`: Directory containing training data

## Installation and Setup

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python3 final_pred.py
   ```

## Team Information

This project was created and developed by Laksh Baweja and team as part of their research into making sign language more accessible through technology. The team focused on creating a practical solution that works reliably in various environments without specialized hardware.

## ABSTRACT

Sign language is one of the oldest and most natural form of language for communication, hence we have come up with a real time method using neural networks for finger spelling based American sign language. Automatic human gesture recognition from camera images is an interesting topic for developing vision. We propose a convolution neural network (CNN) method to recognize hand gestures of human actions from an image captured by camera. The purpose is to recognize hand gestures of human task activities from a camera image. The position of hand and orientation are applied to obtain the training and testing data for the CNN. The hand is first passed through a filter and after the filter is applied where the hand is passed through a classifier which predicts the class of the hand gestures. Then the calibrated images are used to train CNN.

## Introduction

American sign language is a predominant sign language since the only disability D&M people have been communication related and they cannot use spoken languages hence the only way for them to communicate is through sign language. Communication is the process of exchange of thoughts and messages in various ways such as speech, signals, behavior and visuals. Deaf and dumb(D&M) people make use of their hands to express different gestures to express their ideas with other people. Gestures are the nonverbally exchanged messages and these gestures are understood with vision. This nonverbal communication of deaf and dumb people is called sign language.

In our project we basically focus on producing a model which can recognize fingerspelling based hand gestures in order to form a complete word by combining each gesture.

## Requirements

More than 70 million deaf people around the world use sign languages to communicate. Sign language allows them to learn, work, access services, and be included in the communities.

It is hard to make everybody learn the use of sign language with the goal of ensuring that people with disabilities can enjoy their rights on an equal basis with others.

So, the aim is to develop a user-friendly human computer interface (HCI) where the computer understands the American sign language. This project will help the deaf and mute people by making their life easier.

## Objective

To create a computer software and train a model using CNN which takes an image of hand gesture of American Sign Language and shows the output of the particular sign language in text format and converts it into audio format.

## Scope

This system will be beneficial for both deaf/mute people and the people who do not understand sign language. They just need to make sign language gestures and this system will identify what they are trying to say, after which it gives the output in the form of text as well as speech.

## Data Acquisition

The different approaches to acquire data about the hand gesture can be done in the following ways:

1. **Glove-based methods**: These use electromechanical devices to provide exact hand configuration and position. Different glove-based approaches can be used to extract information. However, they are expensive and not user-friendly.

2. **Vision-based methods**: The computer webcam is the input device for observing the information of hands and/or fingers. These methods require only a camera, thus realizing a natural interaction between humans and computers without the use of any extra devices, thereby reducing costs. The main challenge of vision-based hand detection ranges from coping with the large variability of the human hand's appearance due to a huge number of hand movements, to different skin-color possibilities as well as to the variations in viewpoints, scales, and speed of the camera capturing the scene.

## Data Pre-processing and Feature Extraction

In our approach for hand detection, we first detect the hand from the image acquired by webcam using the MediaPipe library. After finding the hand, we get the region of interest (ROI), crop that image and convert it to a gray image using OpenCV. We then apply Gaussian blur and convert the gray image to a binary image using threshold methods.

Initially, we faced challenges with traditional methods as they required a clean background and proper lighting conditions to get accurate results. To overcome these limitations, we developed a novel solution:

1. Detect hand from the frame using MediaPipe
2. Extract hand landmarks
3. Draw and connect those landmarks on a simple white background

This approach tackles the issues of varying backgrounds and lighting conditions because MediaPipe can provide landmark points in almost any environment. We collected 180 skeleton images for alphabets A to Z using this method.

## Convolutional Neural Network (CNN)

CNNs are highly useful in solving computer vision problems. They make use of a filter/kernel to scan through the entire pixel values of the image and set appropriate weights to detect specific features. The architecture includes:

- **Convolutional Layer**: Uses small window size filters (typically 5×5) that scan the input and create 2D activation matrices
- **Pooling Layer**: Decreases the size of activation matrices to reduce learnable parameters
  - Max Pooling: Takes the maximum value from a window (e.g., 2×2)
  - Average Pooling: Takes the average of all values in a window
- **Fully Connected Layer**: Connects all inputs to neurons for final classification

We divided the 26 different alphabets into 8 groups of similar gestures to improve accuracy:

- Group 1: [y, j]
- Group 2: [c, o]
- Group 3: [g, h]
- Group 4: [b, d, f, i, u, v, k, r, w]
- Group 5: [p, q, z]
- Group 6: [a, e, m, n, s, t]

The gesture labels are assigned with probabilities, and the label with the highest probability is considered the predicted label. Using mathematical operations on hand landmarks, we further classify within groups to determine the exact alphabet.

This approach achieved 97% accuracy even with varying backgrounds and lighting conditions, and up to 99% accuracy in optimal conditions.

## Text-to-Speech Translation

The model translates known gestures into words. We use the pyttsx3 library to convert the recognized words into appropriate speech. The text-to-speech output simulates a real-life dialogue and makes the application more accessible.

## Project Requirements

**Hardware Requirement:**

- Webcam

**Software Requirement:**

- Operating System: Windows/Mac/Linux
- Python 3.6 or higher
- Python libraries: OpenCV, NumPy, TensorFlow/Keras, MediaPipe, pyttsx3, Tkinter, PIL

## License

This project is licensed under the MIT License - see the LICENSE file for details.
