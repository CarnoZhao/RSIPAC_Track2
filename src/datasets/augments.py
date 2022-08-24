from albumentations import ImageOnlyTransform
import numpy as np

AUGS = []

try:
    import staintools
    class StainTools(ImageOnlyTransform):
        def __init__(self, always_apply = False, p = 0.5):
            super(StainTools, self).__init__(always_apply=always_apply, p=p)
            self.augmentor = staintools.StainAugmentor(
                method = 'vahadane', 
                sigma1 = 0.2, 
                sigma2 = 0.2)

        def apply(self, img, **params):
            self.augmentor.fit(img)
            img = self.augmentor.pop()
            return np.clip(img.round(), 0, 255).astype(np.uint8)
    AUGS.append(StainTools)
except:
    pass
