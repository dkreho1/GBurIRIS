# GBur-IRIS
An algorithm for computing collision-free convex volumes of configuration space.

## Overview
This repository contains the Python code developed as part of the author's Master's Thesis in the Automatic Control and Electronics programme at the University of Sarajevo (Bosnia and Herzegovina) titled: "Computing collision-free volumes of configuration space". The thesis proposes an algorithm, GBur-IRIS, which computes a set of convex volumes that collectively cover the free configuration space. The Python code from this repository was later ported to C++, which can be found in the [CppGBurIRIS](https://github.com/dzenankreho/CppGBurIRIS) repository.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/dkreho1/GBurIRIS.git
   cd GBurIRIS

2. (Optional) Create and activate a virtual environment:

   ```bash
    python3 -m venv env
    source env/bin/activate

3. Install dependencies:

   ```bash
    pip install -r requirements.txt

4. Run tests to ensure everything is set up correctly:

   ```bash
    pytest ./tests

