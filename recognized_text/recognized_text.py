from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class RecognizedText():
    def __init__(self):
        self.config = Cfg.load_config_from_name('vgg_transformer')
        # config['weights'] = './weights/transformerocr.pth'
        self.config['cnn']['pretrained'] = False
        self.config['device'] = 'cpu'
        self.detector = Predictor(self.config)
    def reconized(self, image):
        result = self.detector.predict(image)
        return result