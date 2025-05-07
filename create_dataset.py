from components.gesture_dataset import gesture_data
import config as cfg

frames = [4, 8]

for frame in frames:
    cfg.FRAMES = frame
    cfg.MODE = "fwdPass"
    X_train, X_test, Y_train, Y_test, n_classes = gesture_data()
    print(f"Frame: {frame}, Shape of X_train: {X_train.shape}, Shape of Y_train: {Y_train.shape}")
    print(f"Frame: {frame}, Shape of X_test: {X_test.shape}, Shape of Y_test: {Y_test.shape}")

    cfg.MODE = "depth"
    X_train, X_test, Y_train, Y_test, n_classes = gesture_data()
    print(f"Frame: {frame}, Shape of X_train: {X_train.shape}, Shape of Y_train: {Y_train.shape}")
    print(f"Frame: {frame}, Shape of X_test: {X_test.shape}, Shape of Y_test: {Y_test.shape}")
