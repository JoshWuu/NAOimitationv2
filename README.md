# NAOimitationv2
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

## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

This project aims to use the Naoqi API and the Mediapipe computer vision library to enable the NAO robot to mimic human movements.

For the implementation of this project, I have relied on the code from https://zenodo.org/records/3935469 for the connection between the robot's movement and the computer vision part. I am continuously enhancing the code to ensure the robot can imitate movements more accurately.

I also relied on Nicolas Renotes tutorial of "AI Pose Estimation with Python and MediaPipe" the video can be found here: https://www.youtube.com/watch?v=05rM3GwxPZM. I used this tutorial to understand how to use the Mediapipe library and how to extract the joint angles from the human pose.
 Due to the fact that mediapipe is not very accurate with the Z axis, it is very hard to get the correct angles from the robot, even using vectors and the dot product. Maybe using a different library or a different approach to get this axis would be better.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This model was built using mostly Python and the NAOqi API and Mediapipe library.
* [![Python][Python.org]][Python-url]
* [![Naoqi][Naoqi.org]][Naoqi-url]
* [![Mediapipe][Mediapipe.org]][Mediapipe-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

If you want to use the code for NAO imitation you can clone the repository and install the necessary dependencies. 
I recommend downloading the python SDK from the Softbank Robotics website to use the NAOqi API. 

<!-- add link -->
### Prerequisites

* Python 2.7 and 3.8 or higher
* Numpy
* OpenCV
* NAOqi API
* Mediapipe



### Installation

You just have to clone the repository and install the necessary dependencies. 
```sh
git clone "https://github.com/Salvatorecoscab/NAOimitationv2.git"
pip install numpy
pip install opencv-python
pip install mediapipe
```
You can find how to install the SDK here: http://doc.aldebaran.com/2-8/dev/python/install_guide.html

We also recommend downloading the Choregraphe software to control the robot in a simulation.

<!-- USAGE EXAMPLES -->
## Usage
To execute the code, the robot must be connected to the same network as the computer, and the following command should be run, in which you have to add the IP of the robot as an argument, or the localhost for the simulation when using Choregraphe:
```bash
python2 server.py "IP of the robot"
```
This will start a server on the computer that will receive joint angles from the client and send them for the robot to move using NAOqi APS. Subsequently, on another terminal, the following command should be run:
```bash
python3 client.py
```
This will initiate a client on the computer responsible for sending the joint angles to the server.
You can also add in the terminal the route of a video file to test the robot movements with a video file with the following command or leave it empty to use the webcam as the video source:
```bash
python3 client.py "route of the video"
```


## Enhancements
* Improve the robot's precision.
* Add more movements.
* Add a GUI to control the robot's movements.

<!-- ROADMAP -->
## Roadmap

- [x] Send data to the robof from the computer
- [x] Visualize the angles of the human pose in the computer
- [ ] Improve the robot's angle precision
- [ ] Control more joints of the robot

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
Salvador Costilla Caballero

Project Link: [https://github.com/Salvatorecoscab/MachineLearningChords](https://github.com/Salvatorecoscab/MachineLearningChords)

[![Product Name Screen Shot][robotWorking   ]](https://example.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments


* [AI Pose Estimation with Python and MediaPipe | Plus AI Gym Tracker Project
](https://www.youtube.com/watch?v=06TE_U21FK4&t=2109s&ab_channel=NicholasRenotte)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[product-screenshot]: Paper/Images/RobotSimulation.png
[robotWorking]: Paper/Images/RobotWorking.png
[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Naoqi.org]: https://img.shields.io/badge/Naoqi-3776AB?style=for-the-badge&logo=naoqi&logoColor=white
[Naoqi-url]: https://developer.softbankrobotics.com/naoqi-2-1
[Mediapipe.org]: https://img.shields.io/badge/Mediapipe-3776AB?style=for-the-badge&logo=mediapipe&logoColor=white
[Mediapipe-url]: https://mediapipe.dev/


