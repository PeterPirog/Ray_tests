
from tensorflow.keras.datasets import mnist
from ray.tune.integration.keras import TuneReportCallback

def train_mnist(config):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf
    import numpy as np

    print('Is cuda available:', tf.test.is_gpu_available())
    batch_size = config['batch_s']
    num_classes = 10
    epochs = 200

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    #define model
    inputs=tf.keras.layers.Input(shape=(28,28,1))  #changed size shape=(28, 28)
    x=tf.keras.layers.BatchNormalization()(inputs)
    #1st conv layer
    x=tf.keras.layers.Conv2D(filters=config["c1_f"],
                             kernel_size=config["c1_ks"],
                             kernel_initializer=config["init"],
                             activation=config["act_f1"])(x)
    x=tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    #2nd conv layer
    x=tf.keras.layers.Conv2D(filters=config["c2_f"],
                             kernel_size=config["c2_ks"],
                             kernel_initializer=config["init"],
                             activation=config["act_f1"])(x)
    x=tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(units=config["hidden"],
                            kernel_initializer=config["init"],
                            activation=config["act_f2"])(x)
    x=tf.keras.layers.Dropout(config["drop"])(x)
    x = tf.keras.layers.BatchNormalization()(x)

    outputs=tf.keras.layers.Dense(units=num_classes,
                                  kernel_initializer=config["init"],
                                  activation="softmax")(x)

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
            "mean_accuracy": "val_accuracy" #optional values ['loss', 'accuracy', 'val_loss', 'val_accuracy']
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
            "mean_accuracy": 0.99,
            "training_iteration": 300
        },
        num_samples=20, #10
        local_dir='./ray_results',
        resources_per_trial={
            "cpu": 8,
            "gpu": 1
        },
        config={
            #"threads": 2,
            'batch_s': tune.choice([8,16, 32, 64,128]),
            "lr": tune.uniform(0.0001, 0.1),
            "hidden": tune.randint(32, 512),
            "drop": tune.uniform(0.01, 0.2),
            "act_f1": tune.choice(["selu"]),
            "act_f2": tune.choice(["selu"]),
            "init": tune.choice(["lecun_normal"]),
            'c1_f':tune.choice([16,32,64]),
            'c2_f':tune.choice([16,32,64]),
            'c1_ks':tune.choice([3,4]),
            'c2_ks':tune.choice([3,4])
        })
    print("Best hyperparameters found were: ", analysis.best_config)