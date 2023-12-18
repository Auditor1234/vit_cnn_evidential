template = "a photo of the "

prompts = [
    # 1
    "Rest",

    # Exercise A(12)
    "Index flexion",
    "Index extension",
    "Middle flexion",
    "Middle extension",
    "Ring flexion",
    "Ring extension",
    "Little finger flexion",
    "Little finger extension",
    "Thumb adduction", # 拇指内收
    "Thumb abduction", # 拇指外转
    "Thumb flexion",
    "Thumb extension",
    
    # Exercise B(17)
    "Thumb up",
    "Extension of index and middle, flexion of the others",
    "Flexion of ring and little finger, extension of the others",
    "Thumb opposing base of little finger",
    "Abduction of all fingers",
    "Fingers flexed together in fist",
    "Pointing index",
    "Adduction of extended fingers",
    "Wrist supination(axis: middle finger)",
    "Wrist pronation(axis: middle finger)",
    "Wrist supination(axis: little finger)",
    "Wrist pronation(axis: little finger)",
    "Wrist flexion",
    "Wrist extension",
    "Wrist radial deviation",
    "Wrist ulnar deviation",
    "Wrist extension with closed hand",

    # Exercise C(23)
    "Large diameter grasp",
    "Small diameter grasp(power grip)",
    "Fixed hook grasp",
    "Index finger extension grasp",
    "Medium wrap",
    "Ring grasp",
    "Prismatic four fingers grasp",
    "Stick grasp",
    "Writing tripod grasp",
    "Power sphere grasp",
    "Three finger sphere grasp",
    "Precesion sphere grasp",
    "Tripod grasp",
    "Prismatic pinch grasp",
    "Tip pinch grasp",
    "Quadop grasp",
    "Lateral grasp",
    "Parallel extension grasp",
    "Extension type grasp",
    "Power disk grasp",
    "Open a bottle with a tripod grasp",
    "Turn a screw(grasp the screwdriver with a stick grasp)",
    "Cut something(grasp the knife with an index finger extension grasp)",

    # Exercise D(9)
    "Flexion of the little finger",
    "Flexion of the ring finger",
    "Flexion of the middle finger",
    "Flexion of the index finger",
    "Abduction of the thumb",
    "Flexion of the thumb",
    "Flexion of index and little finger",
    "Flexion of ring and middle finger",
    "Flexion of index finger and thumb"
]

prompts_sentences = [template + prompt for prompt in prompts]

import os
import torch


def save_results(file, result):
    filename = os.path.basename(file)
    filepath = file.split(filename)[0]
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    with open(file, 'a') as f:
        f.write(result)


def save_model_weight(model, filename='res/best.pt'):
    file = os.path.basename(filename)
    filepath = filename.split(file)[0]
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    torch.save(model.state_dict(), filename)
    print('done')


def setup_seed(seed = 0):
    import torch
    import random
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)



if __name__ == '__main__':
    file = 'res/results.txt'
    save_results(file, '2  hello world!')
