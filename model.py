from imports import *

IMG_SHAPE = (200, 200)
backbone = models.Sequential([
    ResNet101(include_top=False,
              weights='imagenet',
              input_shape=IMG_SHAPE + (3,)),
              layers.Conv2D(1024, 3, 2, activation='relu'),

], name='backbone')

vectoriser = layers.GlobalAveragePooling2D(name='GAP_vectoriser')

regression_head = models.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(4)

], name='regression_head')
