# Project Structure & Instructions

##  Folder Structure

```
├── Datasets/
│   ├── Image Model Dataset/
│   ├── Audio Model Dataset/
│   └── Packet Model Dataset/
│
├── Application + Models/
│   └── app.py  # Main Streamlit application
│   └── (Pretrained model files)
│
├── Code/
│   ├── AudioModel/
│   ├── ImageModel/
│   └── PacketModel/
```

- **Datasets**: Contains training and test datasets for each model (image, audio, and packet).
- **Application + Models**: Contains the main Streamlit application and pre-trained models loaded into the app.
- **Code**: Contains source code for model training, preprocessing, and evaluation for each steganography detection model.

---

##  How to Run the Application

1. **Download** the `Application + Models` folder from the repository.

2. **Open a terminal** and change your directory to the downloaded folder.

   Replace the path below with your actual folder location:

   ```bash
   cd C:/.../Application+Models/
   ```

3. **Launch the Streamlit app** by running:

   ```bash
   streamlit run app.py
   ```

---

##  Dependencies

Make sure you have [Streamlit](https://streamlit.io/) installed. If not, you can install it using:

```bash
pip install streamlit
```
