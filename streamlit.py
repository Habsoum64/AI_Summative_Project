import streamlit as st
import PIL
from ultralytics import YOLO


def main():
    st.title("Object Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        model = YOLO('yolo_model.pt')
        image = PIL.Image.open(uploaded_file)

        results = model.predict(image, show=True)
        results_plot = results[0].plot()[:, :, ::-1]

        st.image(results_plot, caption=(str(len(results[0].boxes)) + " objects detected"))


if __name__ == "__main__":
    main()
