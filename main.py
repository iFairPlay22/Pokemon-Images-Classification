import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import glob
import os
import absl.logging

# Remove warnings & info message...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
absl.logging.set_verbosity(absl.logging.ERROR)

####################################################################################################
###> Programm variables

# ACTIONS
TODO = [ 
    # "from_scratch",
    # "preprocess", 
    "train", 
    # "evaluate", 
    # "test", 
]

# DATASETS
TRAIN_VALIDATION_DATASET = "datasets/tv_dataset/"
TEST_DATASET = "datasets/t_dataset/"
ALLOWED_EXTENSIONS = [ ".jpg", ".jpeg", ".png" ]
IMAGE_SIZE = (180, 180)
NUM_CLASSES = 2

# SAVES
SAVES_PATH = "./saves/"
GRAPHS_PATH = SAVES_PATH + "graphs/"
GRAPHS_TRAINING_HISTORY_FILE_NAME = "training_history.png"
CHECKPOINTS_PATH = SAVES_PATH + "checkpoints/"
CHECKPOINTS_FILE_NAME = "epoch_{epoch}"

# TRAIN
VALIDATION_RATIO = 0.2
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DROPOUT = 0.5
SEED = 1337

# TEST
IMAGE_PATH_TO_TEST = TEST_DATASET + "Negative/00004.jpg"

####################################################################################################
###> Launching the programm

print()
print("Starting...")
print("Actions todo: ", TODO)
print()

####################################################################################################
###> Clean previously saved files

if "from_scratch" in TODO:

    print("Removing files...")
    def removeFilesMatching(path):
        files = glob.glob(path)
        for file in files:
            os.remove(file)
        print("%d files removed matching pattern %s" % (len(files), path))
    removeFilesMatching(CHECKPOINTS_PATH + "*")
    removeFilesMatching(GRAPHS_PATH + "*")

if "train" in TODO:
    def createFolderIfNotExists(folder):
        if not(os.path.isdir(folder)):
            os.makedirs(folder)
    createFolderIfNotExists(CHECKPOINTS_PATH)
    createFolderIfNotExists(GRAPHS_PATH)

####################################################################################################
###> Filter images

if "preprocess" in TODO:

    def isCorrupted(fileimage):

        with open(fileimage, "rb") as fobj:
            if not tf.compat.as_bytes("JFIF") in fobj.peek(10):
                return True

        try:
            with Image.open(fileimage) as img:
                img.verify()
            return False
        except:
            return True

    def removeInvalidImages(folders):

        num_total   = 0
        num_skipped = 0

        # Foreach datasets
        for rootdir in folders:

            # Foreach subdirs
            for f1 in os.listdir(rootdir):
                dir = os.path.join(rootdir, f1)
                if os.path.isdir(dir):

                    # Foreach images
                    for f2 in os.listdir(dir):
                        file = os.path.join(dir, f2)
                        if os.path.isfile(file):

                            if not(
                                any([file.endswith(ext) for ext in ALLOWED_EXTENSIONS]) 
                                    and 
                                not(isCorrupted(file))
                            ):
                                os.remove(file)
                                num_skipped += 1
                                
                            num_total += 1

        print("\nRemove bad formatted files...")
        print("Deleted %d / %d invalid images" % (num_skipped, num_total))

    removeInvalidImages([ TRAIN_VALIDATION_DATASET, TEST_DATASET ])

####################################################################################################
###> Generate the datasets

if "train" in TODO:
    
    print("\nGenerating datasets for training...")

    # Load 80% of the dataset for training
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_VALIDATION_DATASET,
        validation_split=VALIDATION_RATIO,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    # Load 20% of the dataset for validation
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_VALIDATION_DATASET,
        validation_split=VALIDATION_RATIO,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
 
    def visualizeDatasetSample(dataset):
        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(int(labels[i]))
                plt.axis("off")
        plt.show()
    # visualizeDatasetSample(train_ds)

    pass

if "evaluate" in TODO:
    
    print("\nGenerating datasets for evaluation...")

    # Load 100% of the test dataset
    print()
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DATASET,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    )

####################################################################################################
###> Using image data augmentation

if "train" in TODO or "evaluate" in TODO or "test" in TODO:

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
    )

    def visualizeDataAugmentationSample(dataset, data_augmentation):
        plt.figure(figsize=(10, 10))
        for images, _ in dataset.take(1):
            for i in range(9):
                augmented_images = data_augmentation(images)
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(augmented_images[0].numpy().astype("uint8"))
                plt.axis("off")
        plt.show()
    # visualizeDataAugmentationSample(train_ds, data_augmentation)
     
    pass

####################################################################################################
###> Configure the dataset for performance

if "train" in TODO:
    train_ds = train_ds.prefetch(buffer_size=BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=BATCH_SIZE)

####################################################################################################
###> Build a model

if "train" in TODO or "evaluate" in TODO or "test" in TODO:

    def make_model(input_shape, num_classes):
        inputs = keras.Input(shape=input_shape)
        # Image augmentation block
        x = data_augmentation(inputs)

        # Entry block
        x = layers.Rescaling(1.0 / 255)(x)

        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in [128, 256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes

        x = layers.Dropout(DROPOUT)(x)
        outputs = layers.Dense(units, activation=activation)(x)
        return keras.Model(inputs, outputs)

    model = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=NUM_CLASSES)
    keras.utils.plot_model(model, show_shapes=True)

####################################################################################################
###> Train the model

if "train" in TODO or "evaluate" in TODO or "test" in TODO:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

if "train" in TODO:
    
    print("\nTraining the model...")

    callbacks = [ keras.callbacks.ModelCheckpoint(CHECKPOINTS_PATH + CHECKPOINTS_FILE_NAME, save_best_only="True"), ]
    history = model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks, validation_data=val_ds)

    def visualizeLearningHistory(history):
        fig = px.line(
            history.history,
            y=['loss', 'val_loss'],
            labels={'index': "Epoch", 'value': "Loss"},
            title="Training and Validation Loss Over Time"
        )
        plt.savefig(GRAPHS_PATH + GRAPHS_TRAINING_HISTORY_FILE_NAME)

    visualizeLearningHistory(history)

####################################################################################################
###> Load best model

if "evaluate" in TODO or "test" in TODO:

    checkpoints = sorted([ os.path.join(CHECKPOINTS_PATH, file) for file in os.listdir(CHECKPOINTS_PATH) if os.path.isfile(os.path.join(CHECKPOINTS_PATH, file)) ], reverse=True)    
    if 0 < len(checkpoints):
        print("\nLoading checkpoint %s" % checkpoints[0])
        model.load_weights(checkpoints[0])

####################################################################################################
###> Evaluate model

if "evaluate" in TODO:

    def evaluate_model(model, test_ds):
        
        print("\nEvaluating the model...")

        results = model.evaluate(test_ds, verbose=0)
        loss = results[0]
        acc  = results[1]
        
        print("Test Loss: {:.5f}".format(loss))
        print("Test Accuracy: {:.2f}%".format(acc * 100))
    evaluate_model(model, test_ds)

####################################################################################################
###> Test model

if "test" in TODO:

    def test(imgPath):

        print("\nTesting the model...")

        img = keras.preprocessing.image.load_img(imgPath, target_size=IMAGE_SIZE)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)[0]
        print("Result:", [ str(prediction) + "%" for prediction in predictions ])

    test(IMAGE_PATH_TO_TEST)

####################################################################################################
###> Programm end message

print()
print("> Programm exited successfully!")