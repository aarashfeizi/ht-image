import timm


def deit16_224():
    model = timm.create_model('deit_base_patch16_224', pretrained=True)
    model.reset_classifier(0)
    return model

def deit16_small_224():
    model = timm.create_model('deit_small_patch16_224', pretrained=True)
    model.reset_classifier(0)
    return model