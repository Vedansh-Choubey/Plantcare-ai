import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- 1. SETTINGS & PATHS ---
TRAIN_PATH = r"C:\PlantCare-AI\dataset\dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
VAL_PATH = r"C:\PlantCare-AI\dataset\dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"
MODEL_PATH = r"C:\PlantCare-AI\model\plant_disease_model.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# --- 2. DIRECTORY VERIFICATION ---
# Check if model folder exists, if not create it
model_dir = os.path.dirname(MODEL_PATH)
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    print(f"Directory Created: {model_dir}")

# Pre-check:
try:
    with open(MODEL_PATH, 'w') as f:
        f.write("Initial check")
    print("Permission Check: SUCCESS. File can be created.")
except Exception as e:
    print(f"Permission Check: FAILED. Error: {e}")
    print("Tip: Run your IDE/Terminal as Administrator.")

# --- 3. DATA PREPARATION ---
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# --- 4. MODEL BUILDING (MobileNetV2) ---
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 5. CALLBACKS ---

checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# --- 6. TRAINING ---
print("\n--- Training Starting ---")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# --- 7. FINAL SAVING ---
print("\nTraining Finished. Saving final weights...")
model.save(MODEL_PATH)

if os.path.exists(MODEL_PATH):
    size = os.path.getsize(MODEL_PATH) / (1024 * 1024) # Size in MB
    print(f"CONFIRMED: Model saved at {MODEL_PATH}")
    print(f"File Size: {size:.2f} MB")
else:
    print("ALERT: Model file not found in the directory!")