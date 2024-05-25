for inputs, labels in test_dataset:
    bbox_preds = bbox_regressor(inputs, training=False).numpy() 
    bbox_preds = (bbox_preds * (dataset_generator.target_shape * 2)).astype(int)
    imgs = (127 * (inputs + 1)).numpy().astype(np.uint8)
    for idx, img in enumerate(imgs):
        x1, y1, x2, y2 = bbox_preds[idx]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4)
        plt.imshow(img)
        plt.show()
    break