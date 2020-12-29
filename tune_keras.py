
from tensorflow.keras.datasets import mnist
from ray.tune.integration.keras import TuneReportCallback

def train_mnist(config):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf
    print('Is cuda available:', tf.test.is_gpu_available())
    batch_size = 128
    num_classes = 10
    epochs = 12

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(config["hidden"], activation="relu"),
        tf.keras.layers.Dropout(config["dropout"]),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

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
    from ray.tune.schedulers import AsyncHyperBandScheduler
    import tensorflow as tf

    print('Is cuda available external:', tf.test.is_gpu_available())

    mnist.load_data()  # we do this on the driver because it's not threadsafe

    ray.init(num_cpus=8, num_gpus=1)
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20)

    analysis = tune.run(
        train_mnist,
        name="exp",
        scheduler=sched,
        metric="mean_accuracy",
        mode="max",
        stop={
            "mean_accuracy": 0.99,
            "training_iteration": 300
        },
        num_samples=10,
        local_dir='./ray_results',
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        config={
            "threads": 2,
            "lr": tune.uniform(0.001, 0.1),
            "hidden": tune.randint(32, 512),
            "dropout": tune.uniform(0.01, 0.2)
        })
    print("Best hyperparameters found were: ", analysis.best_config)