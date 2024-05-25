from imports import *
from model import *

bbox_regressor = models.Sequential([
    backbone,
    vectoriser,
    regression_head
])

bbox_regressor.summary()
utils.plot_model(bbox_regressor, "localiser.png", show_shapes=False)