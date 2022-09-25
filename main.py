from email.mime import image
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from PIL import Image
import glob
import os
import absl.logging

####################################################################################################
###> Remove warnings & info message...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
absl.logging.set_verbosity(absl.logging.ERROR)

####################################################################################################
###> Programm variables

# ACTIONS
TODO = [ 
    "from_scratch",
    "preprocess", 
    "train", 
    "evaluate", 
    "test", 
]

# DATASETS
DATASET_FOLDER              = "datasets/pokemon/"
TRAIN_VALIDATION_DATASET    = DATASET_FOLDER + "tv_dataset/"
TEST_DATASET                = DATASET_FOLDER + "t_dataset/"
ALLOWED_EXTENSIONS          = [ ".jpg", ".jpeg", ".png" ]
VALIDATION_RATIO            = .2
IMAGE_SIZE                  = (180, 180)

# SAVES
SAVES_PATH                          = "./saves/"
GRAPHS_PATH                         = SAVES_PATH + "graphs/"
GRAPHS_TRAINING_LOSS_FILE_NAME      = "training_loss_history.png"
GRAPHS_TRAINING_ACCURACY_FILE_NAME  = "training_accuracy_history.png"
CHECKPOINTS_PATH                    = SAVES_PATH + "checkpoints/"
CHECKPOINTS_FILE_NAME               = "best_weights"

# TRAIN
TRAINING_PATIENCE   = 5
EPOCHS              = 50
BATCH_SIZE          = 32
LEARNING_RATE       = 1e-3
DROPOUT             = 0.5
SEED                = 1337

# TEST
IMAGE_PATH_TO_TEST  = TEST_DATASET + "Pikachu/00000074.png"

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

        # with open(fileimage, "rb") as fobj:
        #     if not tf.compat.as_bytes("JFIF") in fobj.peek(10):
        #         return True

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
        label_mode="categorical"
    )

    # Load 20% of the dataset for validation
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_VALIDATION_DATASET,
        validation_split=VALIDATION_RATIO,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
 
    def visualizeDatasetSample(dataset):
        plt.figure("Dataset sample", figsize=(10, 10))
        for images, labels in dataset.take(1):
            for i in range(min(9, len(images))):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(int(labels[i]))
                plt.axis("off")
            break
        plt.show()
    # visualizeDatasetSample(train_ds)

    class_names  = train_ds.class_names
    class_number = len(train_ds.class_names)

    pass

if "evaluate" or "test" in TODO:
    
    print("\nGenerating datasets for evaluation...")

    # Load 100% of the test dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DATASET,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    class_names  = test_ds.class_names
    class_number = len(test_ds.class_names)

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
        plt.figure("Data augmentation", figsize=(10, 10))
        for images, _ in dataset.take(1):
            for i in range(min(9, len(images))):
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
        if num_classes == 1:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes

        x = layers.Dropout(DROPOUT)(x)
        outputs = layers.Dense(units, activation=activation)(x)

        return keras.Model(inputs, outputs)

    model = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=class_number)

    # print()
    # print(model.summary())
    # print()
    
    # keras.utils.plot_model(model, show_shapes=True)

    pass

####################################################################################################
###> Train the model

if "train" in TODO or "evaluate" in TODO or "test" in TODO:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

if "train" in TODO:
    
    print("\nTraining the model...")

    callbacks = [ 
        keras.callbacks.ModelCheckpoint(CHECKPOINTS_PATH + CHECKPOINTS_FILE_NAME, save_best_only=True, save_weights_only=True), 
        keras.callbacks.EarlyStopping(monitor="val_loss", restore_best_weights=True, patience=TRAINING_PATIENCE),
    ]
    history = model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks, validation_data=val_ds)

    def visualizeLearningHistory(history):
    
        h = history.history
  
        plt.figure("Loss history")
        plt.plot(h['loss'],         color='red', label='Train loss')
        plt.plot(h['val_loss'],     color='green', label='Val loss')
        plt.legend()
        plt.title('Training and validation loss over the time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(GRAPHS_PATH + GRAPHS_TRAINING_LOSS_FILE_NAME)

        plt.figure("Accuracy history")
        plt.plot(h['accuracy'],     color='red',   label='Train accuracy')
        plt.plot(h['val_accuracy'], color='green', label='Val accuracy')
        plt.legend()
        plt.title('Training and validation accuracy over the time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(GRAPHS_PATH + GRAPHS_TRAINING_ACCURACY_FILE_NAME)

    visualizeLearningHistory(history)

####################################################################################################
###> Load best model

if "evaluate" in TODO or "test" in TODO:

    print("\nLoading checkpoint %s" % (CHECKPOINTS_PATH + CHECKPOINTS_FILE_NAME))
    model.load_weights(CHECKPOINTS_PATH + CHECKPOINTS_FILE_NAME)

####################################################################################################
###> Evaluate model


def loadImage(imgPath):

    loaded_image       = keras.preprocessing.image.load_img(imgPath, target_size=IMAGE_SIZE)
    preprocessed_image = keras.preprocessing.image.img_to_array(loaded_image)

    return loaded_image, preprocessed_image

def extractResults(prediction):

    # Scores
    image_scores = [ 
        { "class_name": class_names[i], "probability": int(prediction[i]*100) } 
        for i in range(len(prediction))
    ]
    image_scores.sort(key=lambda p: p["probability"], reverse=True)

    image_best_score = image_scores[0]
    image_alternative_scores = [ p for p in image_scores[1:] if p["probability"] > 0.2 ]

    return {
        "best_score" : image_best_score,
        "alternative_scores": image_alternative_scores
    }

def makePrediction(model, preprocessed_image):
    prediction = model.predict(tf.expand_dims(preprocessed_image, 0))[0]
    return extractResults(prediction)

def displayImageWithPrediction(name, image, expected=None, predicted=None):

    plt.figure(name)
    plt.imshow(image)

    if expected is not None:    
        plt.text(25, 10, f"Expected: {expected['class_name']} ({expected['probability']})",  bbox=dict(facecolor='grey', alpha=0.5))
    
    if predicted is not None:    
        plt.text(25, 20 if expected is not None else 10, f"Expected: {predicted['class_name']} ({predicted['probability']})", bbox=dict(facecolor='blue', alpha=0.5))
    plt.show()

if "evaluate" in TODO:

    def evaluate_model(model, test_ds):
        
        print("\nEvaluating the model...")

        results = model.evaluate(test_ds, verbose=0)
        loss = results[0]
        acc  = results[1]
        
        print("Test Loss: {:.5f}".format(loss))
        print("Test Accuracy: {:.2f}%".format(acc * 100))

    # evaluate_model(model, test_ds)

    def show_prediction_samples(model, ds):
    
        plt.figure("Prediction samples", figsize=(10, 10))

        for images, labels in ds.take(1):

            for i in range(min(9, len(images))):

                numpy_image = images[i].numpy().astype("uint8")
                extracted_prediction = makePrediction(model, numpy_image)
                predicted_class_name = extracted_prediction['best_score']['class_name']

                numpy_label = labels[i].numpy().astype("float32")
                expected_class_name  = class_names[np.argmax(numpy_label)]

                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(numpy_image)
                plt.title(str(expected_class_name) + " / " + str(predicted_class_name), color="green" if predicted_class_name == expected_class_name else "red")
                plt.axis("off")
            break
        plt.show()

    show_prediction_samples(model, test_ds)

####################################################################################################
###> Test model

if "test" in TODO:

    def test(model, imgPath):

        print("\nTesting the model...")

        loaded_image, preprocessed_image = loadImage(imgPath)

        extracted_prediction = makePrediction(model, preprocessed_image)
        best_score, alternative_best_scores = extracted_prediction['best_score'], extracted_prediction['alternative_scores']

        print("Analizing predictions...")
        print(f"> {best_score['class_name']} with a fiability of {best_score['probability']}%")
            
        if 0 < len(alternative_best_scores):
            print()
            print("Alternative predictions: ")
            for alternative_score in alternative_best_scores:
                print(f"- {alternative_score['class_name']} with a fiability of {alternative_score['probability']}%")

        displayImageWithPrediction("Test " + str(imgPath), image=loaded_image, predicted=best_score)

    test(model, IMAGE_PATH_TO_TEST)

####################################################################################################
###> Programm end message

print()
print("> Programm exited successfully!")