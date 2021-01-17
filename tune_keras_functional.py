from tensorflow.keras.datasets import mnist
from ray.tune.integration.keras import TuneReportCallback


def train_mnist(config):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf
    print('Is cuda available:', tf.test.is_gpu_available())
    batch_size = 128
    num_classes = 10
    epochs = 200

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # define model
    inputs = tf.keras.layers.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten()(inputs)
    # x=tf.keras.layers.LayerNormalization()(x)
    for i in range(config["layers"]):
        x = tf.keras.layers.Dense(units=config["hidden"], activation=config["activation"])(x)
        x = tf.keras.layers.Dropout(config["dropout"])(x)
    outputs = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
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

    ray.init(num_cpus=8, num_gpus=1,
             include_dashboard=True,  # if you use docker use docker run -p 8265:8265 -p 6379:6379
             dashboard_host='0.0.0.0')

    sched = ASHAScheduler(time_attr="training_iteration",
                          max_t=100,
                          grace_period=10,
                          #mode='max',                     #find maximum, do not define here if you define in tune.run
                          reduction_factor=3,
                          brackets=1)

    analysis = tune.run(
        train_mnist,
        name="exp",
        scheduler=sched,
        metric="mean_accuracy",
        mode="max",
        stop={
            "mean_accuracy": 0.99,
            "training_iteration": 200
        },
        num_samples=20,
        #local_dir='/root/ray_results/',  #this is default dir inside docker
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        config={
            # "threads": 2,
            "lr": tune.uniform(0.001, 0.1),
            "hidden": tune.randint(32, 512),
            "dropout": tune.uniform(0.01, 0.2),
            "activation": tune.choice(["relu", "elu"]),
            "layers": tune.choice([1, 2, 3])
        })
    print("Best hyperparameters found were: ", analysis.best_config)
