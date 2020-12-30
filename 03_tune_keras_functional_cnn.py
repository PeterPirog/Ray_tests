
from tensorflow.keras.datasets import mnist
from ray.tune.integration.keras import TuneReportCallback

def train_mnist(config):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf
    import numpy as np

    print('Is cuda available:', tf.test.is_gpu_available())
    batch_size = 128
    num_classes = 10
    epochs = 200

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    #define model
    inputs=tf.keras.layers.Input(shape=(28,28,1))  #changed size shape=(28, 28)
    #1st conv layer
    x=tf.keras.layers.Conv2D(filters=config["conv1_filters"],kernel_size=config["conv1_kernel_size"],activation=config["activation"])(inputs)
    x=tf.keras.layers.MaxPool2D((2,2))(x)
    #2nd conv layer
    x=tf.keras.layers.Conv2D(filters=config["conv2_filters"],kernel_size=config["conv2_kernel_size"],activation=config["activation"])(x)
    x=tf.keras.layers.MaxPool2D((2,2))(x)

    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(units=config["hidden"], activation=config["activation"])(x)
    x=tf.keras.layers.Dropout(config["dropout"])(x)
    outputs=tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)

    model=tf.keras.Model(inputs=inputs,outputs=outputs,name="mnist_conv_model")
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=config["lr"]),
        metrics=["accuracy"])

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneReportCallback({
            "mean_accuracy": "accuracy"
        })])


if __name__ == "__main__":
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    import tensorflow as tf

    print('Is cuda available external:', tf.test.is_gpu_available())

    mnist.load_data()  # we do this on the driver because it's not threadsafe

    ray.init(num_cpus=8, num_gpus=1)
    #sched = AsyncHyperBandScheduler(time_attr="training_iteration", max_t=400, grace_period=20)
    sched = ASHAScheduler(time_attr="training_iteration", max_t=100, grace_period=10)

    analysis = tune.run(
        train_mnist,
        name="exp",
        scheduler=sched,
        metric="mean_accuracy",
        mode="max",
        stop={
            "mean_accuracy": 0.995,
            "training_iteration": 300
        },
        num_samples=100, #10
        local_dir='./ray_results',
        resources_per_trial={
            "cpu": 8,
            "gpu": 1
        },
        config={
            #"threads": 2,
            "lr": tune.uniform(0.0001, 0.1),
            "hidden": tune.randint(32, 512),
            "dropout": tune.uniform(0.01, 0.2),
            "activation": tune.choice(["relu","elu"]),
            'conv1_filters':tune.choice([16,32,64]),
            'conv2_filters':tune.choice([16,32,64]),
            'conv1_kernel_size':tune.choice([(3,3),(4,4)]),
            'conv2_kernel_size':tune.choice([(3,3),(4,4)])
        })
    print("Best hyperparameters found were: ", analysis.best_config)