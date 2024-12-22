import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def Read_image():
     global image, IMG
     file_path = filedialog.askopenfilename(
         title="Select Image",
         filetypes=[("Image Files", "*.jpg;*.png;*.bmp;*.jpeg")],
         initialdir="C:/Users/Nermeen Kamal/PycharmProjects/Show image"
     )

     if not file_path:  # If no file is selected
         messagebox.showinfo("Information", "No file was selected!")
         return

     image = cv2.imread(file_path)

     if image is not None:
         image = cv2.resize(image, (500, 500))  # Resize for display consistency
         IMG = image.copy()  # Optional, if IMG is used for another purpose
         print(f"Image Shape: {image.shape}")  # Display dimensions (height, width, channels)
         messagebox.showinfo("Information", "The image has been read! Now press 'Show'.")
         Show_image_button.config(state=NORMAL)  # Enable the "Show" button
     else:
         messagebox.showinfo("Information", "The image could not be read!")

def Show_image():
     global image
     if image is None:
         messagebox.showinfo("Information", "The image was not read!")
     else:
         update_image_canvas(image)

         image2 = IMG # original image
         image2_data = cv2.resize(image2, (100, 100))

         image_rgb2 = cv2.cvtColor(image2_data, cv2.COLOR_BGR2RGB)
         im_pil2 = Image.fromarray(image_rgb2)
         imgtk2 = ImageTk.PhotoImage(image=im_pil2)
         image_canvas2.image = imgtk2
         image_canvas2.create_image(0, 0, anchor=NW, image=imgtk2)

def RGB_To_Grey():
     global image
     if image is None:
         messagebox.showinfo("Information", "The image was not read!")
     else:
         Grey = np.empty((500, 500), dtype=np.uint8)
         M = np.asarray(image, dtype=np.int32)
         for i in range(len(M)):
          for j in range(len(M[i])):
                 Grey[i][j] = int(M[i][j][0] + M[i][j][1] + M[i][j][2]) // 3
                 M[i][j][0] = Grey[i][j]
                 M[i][j][1] = Grey[i][j]
                 M[i][j][2] = Grey[i][j]
         image = M.astype(np.uint8)
         update_image_canvas(image)

def RGB_To_Binary():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        Binary = np.empty((500, 500), dtype=np.uint8)
        M = np.asarray(image, dtype=np.int32)
        for i in range(len(M)):
            for j in range(len(M[i])):
                if M[i][j][0] > 150:
                    M[i][j][0] = 255
                elif M[i][j][0] <= 150:
                    M[i][j][0] = 0

                if M[i][j][1] > 150:
                    M[i][j][1] = 255
                elif M[i][j][1] <= 150:
                    M[i][j][1] = 0

                if M[i][j][2] > 150:
                    M[i][j][2] = 255
                elif M[i][j][2] <= 150:
                    M[i][j][2] = 0

                Binary[i][j] = (M[i][j][0] + M[i][j][1] + M[i][j][2]) // 3
                M[i][j][0] = Binary[i][j]
                M[i][j][1] = Binary[i][j]
                M[i][j][2] = Binary[i][j]

        image = M.astype(np.uint8)
        update_image_canvas(image)

def Reset():
    global image, IMG
    image = IMG
    update_image_canvas(image)

def Three_Matrix():
     global image
     if(image is None):
         messagebox.showinfo("Information", "The image was not read!")
     else:
         M = np.asarray(image, dtype=np.int32)

         fig = plt.figure(figsize=(3, 9))

         # Red Channel
         ax1 = fig.add_subplot(311)
         ax1.imshow(M[:, :, 0], cmap='Reds', vmin=0, vmax=255)
         ax1.set_title("Red Channel")
         ax1.axis('off')

        # Green Channel
         ax2 = fig.add_subplot(312)
         ax2.imshow(M[:, :, 1], cmap='Greens', vmin=0, vmax=255)
         ax2.set_title("Green Channel")
         ax2.axis('off')

         # Blue Channel
         ax3 = fig.add_subplot(313)
         ax3.imshow(M[:, :, 2], cmap='Blues', vmin=0, vmax=255)
         ax3.set_title("Blue Channel")
         ax3.axis('off')

         update_plot_canvas(fig)

def Add():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        for i in range(len(M)):
            for j in range(len(M[i])):
                if int(M[i][j][0] + 10) > 255:
                    M[i][j][0] = 255
                else:
                    M[i][j][0] = M[i][j][0] + 10

                if int(M[i][j][1] + 10) > 255:
                    M[i][j][1] = 255
                else:
                    M[i][j][1] = M[i][j][1] + 10

                if int(M[i][j][2] + 10) > 255:
                    M[i][j][2] = 255
                else:
                    M[i][j][2] = M[i][j][2] + 10
        image = M.astype(np.uint8)
        update_image_canvas(image)

def Sub():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        for i in range(len(M)):
            for j in range(len(M[i])):
                if int(M[i][j][0] - 10) < 0:
                    M[i][j][0] = 0
                else:
                    M[i][j][0] = M[i][j][0] - 10

                if int(M[i][j][1] - 10) < 0:
                    M[i][j][1] = 0
                else:
                    M[i][j][1] = M[i][j][1] - 10

                if int(M[i][j][2] - 10) < 0:
                    M[i][j][2] = 0
                else:
                    M[i][j][2] = M[i][j][2] - 10
        image = M.astype(np.uint8)
        update_image_canvas(image)
def Multiply():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        for i in range(len(M)):
            for j in range(len(M[i])):
                if int(M[i][j][0] * 5) > 255:
                    M[i][j][0] = 255
                else:
                    M[i][j][0] = M[i][j][0] * 5

                if int(M[i][j][1] * 5) > 255:
                    M[i][j][1] = 255
                else:
                    M[i][j][1] = M[i][j][1] * 5

                if int(M[i][j][2] * 5) > 255:
                    M[i][j][2] = 255
                else:
                    M[i][j][2] = M[i][j][2] * 5

        image = M.astype(np.uint8)
        update_image_canvas(image)

def Division():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        for i in range(len(M)):
            for j in range(len(M[i])):
                if int(M[i][j][0] // 5) < 0:
                    M[i][j][0] = 0
                else:
                    M[i][j][0] = M[i][j][0] // 5

                if int(M[i][j][1] // 5) < 0:
                    M[i][j][1] = 0
                else:
                    M[i][j][1] = M[i][j][1] // 5

                if int(M[i][j][2] // 5) < 0:
                    M[i][j][2] = 0
                else:
                    M[i][j][2] = M[i][j][2] // 5

        image = M.astype(np.uint8)
        update_image_canvas(image)

def Complement():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        for i in range(len(M)):
            for j in range(len(M[i])):
                M[i][j] = 255 - M[i][j]
        image = M.astype(np.uint8)
        update_image_canvas(image)

def Solar():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        for i in range(len(M)-240):
            for j in range(len(M[i])):
                if M[i][j][0] > 160:
                    M[i][j][0] = 100 - M[i][j][0]
                if M[i][j][1] > 160:
                    M[i][j][1] = 100 - M[i][j][1]
                if M[i][j][2] > 160:
                    M[i][j][2] = 100 - M[i][j][2]
        image = M.astype(np.uint8)
        update_image_canvas(image)

def Histo():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = np.zeros(256, dtype=int)

        for i in range(len(grey_image)):
            for j in range(len(grey_image[i])):
                pixel_value = grey_image[i][j]
                hist[pixel_value] += 1

        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.title("Grayscale Histogram")
        plt.bar(range(256), hist, color='black')
        update_histo(hist)

def add_img():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        img_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg;*.png;*.bmp;*.jpeg")],
            initialdir="C:/Users/Nermeen Kamal/PycharmProjects/Show image"
        )
        image2 = cv2.imread(img_path)
        if image2 is not None:
            image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
            result = cv2.add(image, image2)
            image = result
            update_image_canvas(image)
        else:
            messagebox.showinfo("Information", "The image was not read!")

def sub_img():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        imgg_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg;*.png;*.bmp;*.jpeg")],
            initialdir="C:/Users/Nermeen Kamal/PycharmProjects/Show image"
        )
        image2 = cv2.imread(imgg_path)
        if image2 is not None:
            image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))
            result = cv2.subtract(image.astype(np.int32), image2.astype(np.int32))
            result = np.clip(result, 0, 255).astype(np.uint8)

            image = result
            update_image_canvas(image)
        else:
            messagebox.showinfo("Information", "The image was not read!")

def swapRB():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        Temp = M[:, :, 0].copy()
        M[:, :, 0] = M[:, :, 2]
        M[:, :, 2] = Temp
        image = M.astype(np.uint8)
        update_image_canvas(image)

def swapRG():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        Temp = M[:, :, 0].copy()
        M[:, :, 0] = M[:, :, 1]
        M[:, :, 1] = Temp
        image = M.astype(np.uint8)
        update_image_canvas(image)

def swapBG():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        Temp = M[:, :, 1].copy()
        M[:, :, 1] = M[:, :, 2]
        M[:, :, 2] = Temp
        image = M.astype(np.uint8)
        update_image_canvas(image)

def eliminationR():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        M[:, :, 1] = 0
        M[:, :, 0] = 0
        R = M.astype(np.uint8)
        update_image_canvas(R)

def eliminationG():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        M[:, :, 0] = 0
        M[:, :, 2] = 0
        G = M.astype(np.uint8)
        update_image_canvas(G)

def eliminationB():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        M[:, :, 1] = 0
        M[:, :, 2] = 0
        B = M.astype(np.uint8)
        update_image_canvas(B)

def eliminationBandG():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        M[:, :, 2] = 0
        BandG = M.astype(np.uint8)
        update_image_canvas(BandG)

def eliminationRandG():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        M[:, :, 0] = 0
        RandG = M.astype(np.uint8)
        update_image_canvas(RandG)

def eliminationBandR():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.int32)
        M[:, :, 1] = 0
        BandR = M.astype(np.uint8)
        update_image_canvas(BandR)

def Save():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")],
                                            initialdir="C:/Users/Nermeen Kamal/PycharmProjects/Show image/save")

        cv2.imwrite(path, image)
        messagebox.showinfo("Saved", "The image was saved successfully!")

def HistoEqualization():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.uint8)

        if len(M.shape) == 2:
            M = cv2.equalizeHist(M)
        elif len(M.shape) == 3 and M.shape[2] == 3:
            for channel in range(3):
                M[:, :, channel] = cv2.equalizeHist(M[:, :, channel])
        else:
            messagebox.showinfo("Error", "Unsupported image format!")
        image = M.astype(np.uint8)
        update_image_canvas(image)

def HistoStretching():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
    else:
        M = np.asarray(image, dtype=np.float32)
        min_val = M.min()
        max_val = M.max()

        for i in range(len(M)):
            for j in range(len(M[i])):
                if len(M[i][j]) == 3:  # RGB image
                    for k in range(3):  # Iterate over R, G, B channels
                        new_value = 255 * (M[i][j][k] - min_val) / (max_val - min_val)
                        M[i][j][k] = np.clip(new_value, 0, 255)

                else:  # Grayscale image
                    new_value = 255 * (M[i][j] - min_val) / (max_val - min_val)
                    M[i][j] = np.clip(new_value, 0, 255)

        image = M.astype(np.uint8)
        update_image_canvas(image)

def Mean():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
        return

    if len(image.shape) == 3 and image.shape[2] == 3:
        channels = [image[:, :, i] for i in range(3)]
        filtered_channels = []

        kernel_size = 3
        padding = kernel_size // 2

        for channel in channels:
            padded_channel = np.pad(channel, pad_width=padding, mode='constant', constant_values=0)
            filtered_channel = np.zeros_like(channel)

            for i in range(channel.shape[0]):
                for j in range(channel.shape[1]):
                    region = padded_channel[i:i + kernel_size, j:j + kernel_size]
                    filtered_channel[i, j] = np.mean(region)

            filtered_channels.append(filtered_channel)

        filtered_image = np.stack(filtered_channels, axis=2)

    else:
        kernel_size = 3
        padding = kernel_size // 2

        padded_image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
        filtered_image = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                filtered_image[i, j] = np.mean(region)

    image = filtered_image.astype(np.uint8)
    update_image_canvas(image)


def Median():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
        return

    kernel_size = 3
    offset = kernel_size // 2

    filtered_image = [[[0 for _ in range(3)] for _ in range(500)] for _ in range(500)]

    for i in range(offset, 500 - offset):
        for j in range(offset, 500 - offset):
            for c in range(3):  # (R, G, B)
                neighborhood = []
                for ki in range(-offset, offset + 1):
                    for kj in range(-offset, offset + 1):
                        neighborhood.append(image[i + ki][j + kj][c])

                count = 0
                for _ in neighborhood:
                    count += 1

                #  Bubble Sort
                for x in range(count - 1):
                    for y in range(count - x - 1):
                        if neighborhood[y] > neighborhood[y + 1]:
                            neighborhood[y], neighborhood[y + 1] = neighborhood[y + 1], neighborhood[y]

                median_value = neighborhood[count // 2]
                filtered_image[i][j][c] = median_value

    image = filtered_image
    update_image_canvas(image)


def Laplacian():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
        return

    height, width, channels = 500, 500, 3

    filtered_image = [[[0 for _ in range(channels)] for _ in range(width)] for _ in range(height)]

    kernel = [
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ]

    kernel_size = 3
    offset = kernel_size // 2

    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            for c in range(channels):
                laplace_sum = 0
                for ki in range(-offset, offset + 1):
                    for kj in range(-offset, offset + 1):
                        pixel_value = int(image[i + ki][j + kj][c])  # تحويل إلى int لمنع overflow
                        kernel_value = kernel[ki + offset][kj + offset]
                        laplace_sum += pixel_value * kernel_value

                filtered_value = max(0, min(255, laplace_sum))
                filtered_image[i][j][c] = filtered_value

    image = np.array(filtered_image, dtype=np.uint8)
    update_image_canvas(image)

def Average():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
        return

    height, width, channels = 500, 500, 3

    filtered_image = [[[0 for _ in range(channels)] for _ in range(width)] for _ in range(height)]

    kernel_size = 3
    kernel_area = kernel_size * kernel_size
    offset = kernel_size // 2

    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            for c in range(channels):
                pixel_sum = 0
                for ki in range(-offset, offset + 1):
                    for kj in range(-offset, offset + 1):
                        pixel_value = int(image[i + ki][j + kj][c])  # تحويل إلى int لمنع overflow
                        pixel_sum += pixel_value

                average_value = pixel_sum // kernel_area

                filtered_image[i][j][c] = max(0, min(255, average_value))

    image = np.array(filtered_image, dtype=np.uint8)
    update_image_canvas(image)

def Gaussian():
    global image
    if image is None:
        messagebox.showinfo("Information", "The image was not read!")
        return
    M = np.asarray(image, dtype=np.int16)

    for i in range(len(M)):
        for j in range(len(M[i])):
            noise = np.random.randint(-20, 20)

            M[i][j] = M[i][j] + noise

            if M[i][j][0] > 255:
                M[i][j][0] = 255
            elif M[i][j][0] < 0:
                M[i][j][0] = 0

            if M[i][j][1] > 255:
                M[i][j][1] = 255
            elif M[i][j][1] < 0:
                M[i][j][1] = 0

            if M[i][j][2] > 255:
                M[i][j][2] = 255
            elif M[i][j][2] < 0:
                M[i][j][2] = 0

    image = M.astype(np.uint8)
    update_image_canvas(image)


def dilate_image():
    global image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary_image, kernel, iterations=1)
    update_image_canvas(dilated)

def erode_image():
    global image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(binary_image, kernel, iterations=1)
    update_image_canvas(eroded)

def open_image():
    global image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    update_image_canvas(opened)

def close_image():
    global image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    update_image_canvas(closed)



def update_image_canvas(image_data):
    image_array = np.array(image_data, dtype=np.uint8)
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image_rgb)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    image_canvas.image = imgtk
    image_canvas.create_image(0, 0, anchor=NW, image=imgtk)


def update_plot_canvas(fig):
    for widget in plot_canvas.winfo_children():
        widget.destroy()
    plot_canvas_agg = FigureCanvasTkAgg(fig, master=plot_canvas)
    plot_canvas_agg.draw()
    plot_canvas_agg.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

def update_histo(hist):
    for widget in hist_canvas.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots(figsize=(4, 2))  # Adjust size to fit under the image
    ax.bar(range(len(hist)), hist, color='black')  # Plot the histogram
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram")
    fig.tight_layout()

    # Display the figure on hist_canvas
    hist_canvas_agg = FigureCanvasTkAgg(fig, master=hist_canvas)
    hist_canvas_agg.draw()
    hist_canvas_agg.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)


frame = Tk()
frame.title("Image Processing System")
frame.geometry("1800x900")
frame.resizable(True, True)
frame.configure(background="#4e2e2e")

label = Label(frame, text="Image Processing System", bg="#4e2e2e", fg="white", font=("Arial", 30, "bold"), pady=2)
label.place(x=470, y=10)

 # Canvas for displaying images
image_canvas = Canvas(frame, width=500, height=500, background="#7e3e3e")
image_canvas.place(x=400, y=75)

image_canvas2 = Canvas(frame, width=100, height=100, background="#7e3e3e")
image_canvas2.place(x=298, y=75)
label2 = Label(frame, text="Original", bg="#4e2e2e",fg="white", font=("Arial", 13, "bold"))
label2.place(x=315, y=186)

 # Canvas for displaying plots
plot_canvas = Frame(frame)
plot_canvas.place(x=910, y=78, width=300, height=500)  # Positioned next to the image canvas

hist_canvas = Frame(frame)
hist_canvas.place(x=490, y=595, width=650, height=184)  # Positioned under the image canvas

 # Canvas options
i = Canvas(frame, width=180, height=1202, background="#7e3e3e", highlightthickness=1)
i.place(x=-1, y=-1)
i2 = Canvas(frame, width=170, height=1202, background="#7e3e3e", highlightthickness=1)
i2.place(x=1366, y=-1)

Read_image_button = Button(frame, text="Read image", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Read_image)
Show_image_button = Button(frame, text="Show image", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Show_image)
Save_button = Button(frame, text="Save image", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Save)
RGB_To_Grey_button = Button(frame, text="Image To Grey", fg="white", bg="gray", font=("Arial", 10, "bold"), command=RGB_To_Grey)
Reset_button = Button(frame, text="Reset", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Reset)
Add_button = Button(frame, text="Addition", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Add)
Sub_button = Button(frame, text="Subtraction", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Sub)
Mult_button = Button(frame, text="Multiplication", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Multiply)
Division_button = Button(frame, text="Division", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Division)
Complement_button = Button(frame, text="Complement", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Complement)
Solar_button = Button(frame, text="Solarization", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Solar)
Add_img_button = Button(frame, text="Add image", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=add_img)
Sub_img_button = Button(frame, text="Sub image", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=sub_img)

swapRB_button = Button(frame, text="Swap Blue & Red", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=swapRB)
swapRG_button = Button(frame, text="Swap Green & Blue", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=swapRG)
swapBG_button = Button(frame, text="Swap Green & Red", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=swapBG)

R_button = Button(frame, text="R", fg="red", bg="#b8b8b8", font=("Arial", 10, "bold"), command=eliminationR)
G_button = Button(frame, text="G", fg="green", bg="#b8b8b8", font=("Arial", 10, "bold"), command=eliminationG)
B_button = Button(frame, text="B", fg="blue", bg="#b8b8b8", font=("Arial", 10, "bold"), command=eliminationB)
BG_button = Button(frame, text="BG", fg="cyan", bg="#7a7a7a", font=("Arial", 10, "bold"), command=eliminationBandG)
RG_button = Button(frame, text="RG", fg="yellow", bg="#7a7a7a", font=("Arial", 10, "bold"), command=eliminationRandG)
BR_button = Button(frame, text="RB", fg="magenta", bg="#7a7a7a", font=("Arial", 10, "bold"), command=eliminationBandR)

Histo_button = Button(frame, text="Histogram", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Histo)
HistoStretching_Button = Button(frame, text="Stretching", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=HistoStretching)
HistoEqualization_Button = Button(frame, text="Equalization", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=HistoEqualization)
Three_Matrix_button = Button(frame, text="Separate Channels", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Three_Matrix)

Mean_button = Button(frame, text="Mean Filter", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Mean)
Median_button = Button(frame, text="Median Filter", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Median)
Laplacian_button = Button(frame, text="Laplacian Filter", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Laplacian)
Gaussian_Noise_button = Button(frame, text="Gaussian Noise", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Gaussian)
Average_button = Button(frame, text="Average filter", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=Average)

dilate_image_button = Button(frame, text="Dilation filter", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=dilate_image)
erode_image_button = Button(frame, text="Erosion filter", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=erode_image)
open_image_button = Button(frame, text="Opening filter", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=open_image)
close_image_button = Button(frame, text="Closing filter", fg="white", bg="#7e3e3e", font=("Arial", 10, "bold"), command=close_image)

RGB_To_Binary_button = Button(frame, text="Image To Binary", fg="white", bg="black", font=("Arial", 10, "bold"), command=RGB_To_Binary)


Read_image_button.place(x=30, y=20)
Show_image_button.place(x=30, y=60)
Save_button.place(x=30, y=100)
RGB_To_Grey_button.place(x=30, y=200)
Reset_button.place(x=30, y=140)
Add_button.place(x=30, y=260)
Sub_button.place(x=30, y=300)
Mult_button.place(x=30, y=340)
Division_button.place(x=30, y=380)
Complement_button.place(x=30, y=430)
Solar_button.place(x=30, y=470)
Add_img_button.place(x=30, y=525)
Sub_img_button.place(x=30, y=565)
swapRB_button.place(x=30, y=620)
swapRG_button.place(x=30, y=660)
swapBG_button.place(x=30, y=700)
R_button.place(x=30, y=740)
G_button.place(x=60, y=740)
B_button.place(x=90, y=740)
BG_button.place(x=30, y=780)
RG_button.place(x=70, y=780)
BR_button.place(x=110, y=780)
Histo_button.place(x=1410, y=20)
HistoStretching_Button.place(x=1410, y=60)
HistoEqualization_Button.place(x=1410, y=100)
Three_Matrix_button.place(x=1390, y=140)
Mean_button.place(x=1390, y=200)
Median_button.place(x=1390, y=240)
Average_button.place(x=1390, y=300)
Laplacian_button.place(x=1390, y=340)
Gaussian_Noise_button.place(x=1390, y=380)
dilate_image_button.place(x=1390, y=440)
erode_image_button.place(x=1390, y=480)
open_image_button.place(x=1390, y=520)
close_image_button.place(x=1390, y=560)
RGB_To_Binary_button.place(x=1390, y=620)

frame.mainloop()