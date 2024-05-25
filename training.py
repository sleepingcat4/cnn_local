EPOCHS = 100
LEARNING_RATE = 0.0003

from imports import *
from build import *
from metrics import *

lr_scheduler = optimizers.schedules.ExponentialDecay(LEARNING_RATE, 3600, 0.8)
optimizer = optimizers.Adam(learning_rate=lr_scheduler)


for epoch in range(EPOCHS):
    train_losses, train_ious = np.array([]), np.array([])
    for step, (inputs, labels) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            preds = bbox_regressor(inputs, training=True)
            loss, iou = criteron(labels, preds)
        
        grads = tape.gradient(loss, bbox_regressor.trainable_weights)
        optimizer.apply_gradients(zip(grads, bbox_regressor.trainable_weights))

        loss_value = tf.math.reduce_mean(loss).numpy()
        train_losses = np.hstack([train_losses, loss_value])

        iou_value = tf.math.reduce_mean(iou).numpy()
        train_ious = np.hstack([train_ious, iou_value])

        print('Training Loss :  %f'%(step + 1, math.ceil(train_max / BATCH_SIZE), 
              loss_value), end='')
        
        tr_lss, tr_iou = np.mean(train_losses), np.mean(train_ious)
        print('Train loss : %f -- Train Average IOU : %f' % (epoch, EPOCHS, tr_lss, tr_iou))
        print()