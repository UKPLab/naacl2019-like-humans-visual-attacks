import os

def get_last_model_path(save_dir, modelinit):
    """
    Get the last saved model
    :param save_dir: Directory where models are saved
    :return: the full path of the last model, next epoch (if we will continue training)
    """
    nexte = 2
    model_name = modelinit
    for file in os.listdir(save_dir):
        if file.endswith(".h5") and file.__contains__("_"):
            ce = int(file.split('_')[1][:-3])+1
            if ce>nexte:
                nexte = ce
                model_name = file
    full_path = os.path.join(save_dir, model_name)
    return full_path, nexte

def remove_except_last_model(save_dir, modelinit):
    """
    Remove all model files except the last one
    :param save_dir: Directory where models are saved
    :return:
    """
    fullp, nexte = get_last_model_path(save_dir, modelinit)
    laste = nexte
    for file in os.listdir(save_dir):
        if file.endswith(".h5") and file.__contains__("_"):
            ce = int(file.split('_')[1][:-3])+1
            if ce<laste:
                model_name = file
                full_path = os.path.join(save_dir, model_name)
                os.remove(full_path)
    return