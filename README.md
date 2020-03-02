# Overview

This project is to recognize person face and detect his age and gender on Jetson Nano. Two Raspberry Pi cameras 
connected to Jetson Nano through CSI port capture human faces and send his information(age, gender, time stamp, id, 
face coordinate) to server using REST API.

Also, Jetson Nano board remembers human faces captured for 30 minutes so that it can assign the same id to the 
same person for that time. Then it recognizes the events of human's moving in and out camera, and every time 
human looks forward the camera, it estimates his age and gender.



## Project Structure

- source
    
    The main source code to detect face and estimate age and gender is contained.
    
- utils

    Tools concerned with image processing and file management are contained.

- app

    This is the main execution file.

- requirements
    
    All the libraries to execute project are inserted.

- settings

    Several settings are conducted in this file

## Project Install

This project is performed on Jetson Nano with Balena OS, which is implemented by Docker.

- Environment

   Balena OS 
    
- Python 3.6 environment

- Download zip file which contains tbz2 files for setup cuda cudn on Jetson Nano and opencv files from 
https://drive.google.com/file/d/1yiIcHe8gjH-5ji6tzz5dlblQM7EjuunP/view?usp=sharing and extract it in this project 
directory.

## Project Execution

- Pull this project from Balena application
