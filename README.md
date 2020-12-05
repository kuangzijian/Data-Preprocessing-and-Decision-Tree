
<!-- PROJECT LOGO
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</p>
-->


<!-- TABLE OF CONTENTS 
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)
-->


<!-- ABOUT THE PROJECT -->
## Data Preprocessing and Decision Tree

This project focuses on topics:
* Data Preprocessing
* Feature Selection
* Building a Decision Tree using Information Gain

### Built With
* [Pytorch](https://github.com/pytorch)

### Prerequisites
```sh
1. Clone the repo
2. pip install -r requirements.txt
```

### Q1 Do feature scaling by these two approaches and then explain what the reasons is for doing feature scaling:

```
The reasons for doing feature scaling is to normalize the data within a small, specified range, so that it can speed up the calculations in machine learning algorithm.
```

### Q2 Explain the two feature selection techniques, interpret the output, and discuss which technique you will use for a large dataset and why?

```
The first technique is a filter method, it measures the relevance of features by their correlation with the outcome variable; the second technique is a wrapper method which measures how useful the feature is by training with a machine learning algorithm on it.
Although wrapper methods are much slower and more computationally expensive than filter methods, I will still use the wrapper method on large dataset, because it can always provide the best subset of features. We also do not need to worry about overfitting problem by using wrapper method if the number of observations is sufficient.
```

### Q3 Building a Decision Tree using Information Gain

![](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-MM811-2020-Assignment-1/blob/main/images/Decision%20Tree.png)
```
H(D)= -1/2* LOG(1/2,2)-1/2*LOG(1/2,2) = 1
H(D|A='X') = 3/4*(-2/3*LOG(2/3,2)-1/3*LOG(1/3,2))+1/4*(-0/1*LOG(0/1,2)-1/1*LOG(1/1,2)) = 0.689
H(D|A='Y') = 2/4*(-2/2*LOG(2/2,2)-0/2*LOG(0/2,2))+2/4*(-0/2*LOG(0/2,2)-2/2*LOG(2/2,2)) = 0
H(D|A='Z') = 2/4*(-1/2*LOG(1/2,2)-1/2*LOG(1/2,2))+2/4*(-1/2*LOG(1/2,2)-1/2*LOG(1/2,2)) = 1
g(D, A='X') = H(D) - H(D|A='X') = 0.311
g(D, A='Y') = H(D) - H(D|A='Y') = 1
g(D, A='Z') = H(D) - H(D|A='Z') = 0
Since attribute “Y” has highest IG (0.311), we use “Y” as root node.
```
![Generated usking graphviz lib](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-MM811-2020-Assignment-1/blob/main/images/diabetes.png)

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

## References


