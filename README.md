# Object Detection App

This is an object detection app built with Streamlit. It allows you to upload an image and performs object detection on it using OpenCV. This readme file will guide you on how to set up and host the application either on a local server or on the cloud.

## Installation

1. Clone the repository:

```
git clone <repository_url>
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

To run the app on a local server, follow these steps:

1. Open a terminal or command prompt and navigate to the project directory.
2. Run the following command to start the app:

```
streamlit run app.py
```

3. The app will launch in your default web browser.
4. Click on the "Choose an image..." button to upload an image file (supported formats: jpg, png, jpeg).
5. The app will perform object detection on the uploaded image.
6. Detected objects will be highlighted with rectangles.
7. The modified image with detected objects will be displayed on the app.

## Hosting on the Cloud

To host the app on the cloud, you can use platforms like Heroku or AWS. Here's a general guide on hosting the app on Heroku:

1. Sign up for a Heroku account at [https://signup.heroku.com/](https://signup.heroku.com/) if you don't have one.
2. Install the Heroku CLI by following the instructions at [https://devcenter.heroku.com/articles/heroku-cli](https://devcenter.heroku.com/articles/heroku-cli).
3. Log in to your Heroku account using the following command:

```
heroku login
```

4. In the project directory, create a new Heroku app:

```
heroku create <app_name>
```

5. Commit your changes to Git:

```
git add .
git commit -m "Initial commit"
```

6. Deploy the app to Heroku:

```
git push heroku main
```

7. Once the deployment is complete, run the following command to open the app in your default browser:

```
heroku open
```

8. The app should now be hosted on the Heroku platform, and you can access it through the provided URL.

Note: This is a general guide, and the specific steps may vary depending on the hosting platform you choose. Please refer to their documentation for more detailed instructions.

## Customization

You can customize the object detection logic by modifying the `perform_object_detection` function in the `app.py` file. Replace the existing logic with your own object detection algorithm.

## Dependencies

The app uses the following dependencies:

- streamlit: Used to build the web app
- OpenCV (cv2): Used for image processing and object detection
- numpy: Used for array manipulation

You can find the specific versions of these dependencies in the `requirements.txt` file.

## This code is not complete
We did not load our model to the streamlit app. Therefore, this serves only as a demonsttration using an in-built model for face recognition. 
