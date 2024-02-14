# MachineLearningChords

<a name="readme-top"></a>




<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

When a person starts learning how to play guitar, they often start learning some chords. That is why I decided to create a machine lerning model that from a video, an image or just using the webcam, it can recognize the chords that are being played. 
I have already trained the model with 11 different chords, and I am planning to add more chords in the future. The chords that the model can recognize are: A, Am, Bm, B7, C, D, D7, Dm, Em, F, and G.
This chord detection application might be useful in the future to develop more complex applications, such as a guitar teacher.
This is a project that I am doing for my Machine Learning lecture at Hof University of Applied Sciences.
If you want to see the images that I used to train the model, you can find them in the next links:
* [dataset with augmnentation](https://app.roboflow.com/school-sps5k/chorddetection2.2/1)
* [dataset without augmentation](https://drive.google.com/drive/folders/1bDZHMe4fbbrXj0HvlKFn1YTyW3PKdSHg?usp=drive_link)

If you want to see the explanation of the code for the application or how I trained the model, you can find it in the following links:
* [Code for the application](CODE.md)
* [Training the model](TRAIN.md)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This model was built using mostly Python, Roboflow, YOLOv8, OpenCV, and PyTorch. 

* [![Python][Python.org]][Python-url]
* [![Roboflow][Roboflow.com]][Roboflow-url]
* [![YOLOv8][YOLOv8.com]][YOLOv8-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

If you want use one of the two models that I have already trained for the application that recognizes the chords you can follow the instructions below.

<!-- add link -->
### Prerequisites
* Python 3.8 or higher
* OpenCV
* YOLO
* Download one of the models from the following links:
    * [Version 1](https://drive.google.com/file/d/1UH5IGShcNTGRWJmEr_HkUWbeh4Hg4eXh/view?usp=drive_link)
    * [Version 2](https://drive.google.com/file/d/1dkf01HCFlMJX_NJqrYd8zBbyCNi5ZZN8/view?usp=drive_link)

* Change the name of the model to "best.pt" in "ChordDetection.py" and put the model in the models folder.


### Installation

You just have to clone the repository and install the necessary dependencies. 
```sh
git clone
pip install opencv-python
pip install ultralytics
```
once the dependencies are installed, you just have to download one of the models from the links above and put it in the same folder as the ChordDetection.py file.

<!-- USAGE EXAMPLES -->
## Usage

To start the chord detection application, once you have cloned the repository and installed the necessary dependencies, you just have to run the following in the terminal:
```sh
python ChordDetection.py
```
This will open the webcam and start detecting the chords that the camera can see. If you want to use a video, you can use the following command:
```sh
python ChordDetection.py --source "path to the video or image"
```




<!-- ROADMAP -->
## Roadmap

- [x] detect chords from webcam and video
- [ ] Add more chords to the model
- [ ] Train a new model with more chords
- [ ] Create a GUI for the application

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing
If you have any ideas on how to improve the application, feel free to contribute. Here are the steps to do so:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>






<!-- CONTACT -->
## Contact

Project Link: [https://github.com/Salvatorecoscab/MachineLearningChords](https://github.com/Salvatorecoscab/MachineLearningChords)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments


* [Train YOLOv8 on Custom Dataset | Sign Language Alphabets Detection and Recognition using YOLOv8](https://www.youtube.com/watch?v=-UoSr9Z_Bg0&ab_channel=MuhammadMoin)
* [YOLOV8: Object Detection Annotation | Annotate Custom Data using Roboflow](https://www.youtube.com/watch?v=qn96xC3LV2Y&ab_channel=ArtificiallyIntelligent)
* [YOLOv8](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[product-screenshot]: ProjectImages/ImageTest.png
[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Roboflow.com]: https://img.shields.io/badge/Roboflow-4A4A55?style=for-the-badge&logo=roboflow&logoColor=FF3E00
[Roboflow-url]: https://roboflow.com/
[YOLOv8.com]: https://img.shields.io/badge/YOLOv8-4A4A55?style=for-the-badge&logo=yolov8&logoColor=FF3E00
<!-- ultralytics -->
[YOLOv8-url]: https://github.com/ultralytics/ultralytics?tab=readme-ov-file
