# Image Processing Application

## Overview
This Python application provides a graphical user interface (GUI) for performing various image processing operations. Users can upload, manipulate, and save images using a range of techniques, including grayscale conversion, binary conversion, histogram equalization, and more.

## Features
- **Image Loading**: Load and display images from local storage.
- **Basic Image Operations**:
  - Convert to grayscale
  - Convert to binary
  - Add or subtract brightness
  - Complement and solarization
  - Histogram equalization and stretching
- **Color Channel Manipulations**:
  - Swap or eliminate RGB channels
  - Display individual color channels
- **Arithmetic Operations on Images**:
  - Add or subtract one image from another
  - Multiply or divide pixel values
- **Save Processed Images**: Save the manipulated image to your computer.

## Technologies Used
- **Python Libraries**:
  - `OpenCV` for image processing
  - `Tkinter` for the GUI
  - `NumPy` for numerical computations
  - `Pillow` for image handling in the GUI
  - `Matplotlib` for plotting histograms and color channels

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/image-processing-app.git
   cd image-processing-app
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use
1. Run the application:
   ```bash
   python app.py
   ```
2. Use the GUI to:
   - Load an image using the "Read" button.
   - Perform various operations by clicking the corresponding buttons (e.g., "Convert to Grayscale").
   - Save the processed image using the "Save" button.

## Functionalities
### 1. Image Loading
- **`Read_image()`**: Allows users to select an image file to load into the application.

### 2. Image Display
- **`Show_image()`**: Displays the currently loaded image.

### 3. Image Processing Operations
- **`RGB_To_Grey()`**: Converts the image to grayscale.
- **`RGB_To_Binary()`**: Converts the image to binary format.
- **`Add()`**: Increases pixel brightness by 10.
- **`Sub()`**: Decreases pixel brightness by 10.
- **`Multiply()`**: Multiplies pixel values by 5.
- **`Division()`**: Divides pixel values by 5.
- **`Complement()`**: Computes the complement of the image.
- **`Solar()`**: Applies a solarization effect.

### 4. Color Manipulations
- **`swapRB()`, `swapRG()`, `swapBG()`**: Swaps specified RGB channels.
- **`eliminationR()`, `eliminationG()`, `eliminationB()`**: Eliminates specific RGB channels.

### 5. Histogram Operations
- **`Histo()`**: Displays the grayscale histogram.
- **`HistoEqualization()`**: Performs histogram equalization.
- **`HistoStretching()`**: Stretches the histogram to enhance contrast.

### 6. Image Arithmetic
- **`add_img()`**: Adds another image to the current image.
- **`sub_img()`**: Subtracts another image from the current image.

### 7. Reset and Save
- **`Reset()`**: Resets the image to its original state.
- **`Save()`**: Saves the current image to a specified location.

## File Structure
```
image-processing-app/
|-- app.py               # Main application file
|-- README.md            # Project documentation
|-- requirements.txt     # List of dependencies
```

## Dependencies
- Python 3.6+
- OpenCV
- NumPy
- Tkinter (bundled with Python)
- Pillow
- Matplotlib

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
Nermeen Kamal  
Faculty of Computer Science and Information Technology  
Ahram Canadian University
