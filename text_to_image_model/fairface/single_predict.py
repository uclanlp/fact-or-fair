import os
import dlib
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision

current_dir = os.path.dirname(os.path.abspath(__file__))

def load_models():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the 7-class model and map it to the available device
    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(
        torch.load(os.path.join(current_dir, 'fair_face_models/res34_fair_align_multi_7_20190809.pt'), map_location=device))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    return model_fair_7, device


def detect_and_predict_single_image(image_path):
    # Load Dlib models
    cnn_face_detector = dlib.cnn_face_detection_model_v1(os.path.join(current_dir, './dlib_models/mmod_human_face_detector.dat'))
    sp = dlib.shape_predictor(os.path.join(current_dir, './dlib_models/shape_predictor_5_face_landmarks.dat'))

    # Load and resize the image
    img = dlib.load_rgb_image(image_path)
    old_height, old_width, _ = img.shape
    default_max_size = 800
    if old_width > old_height:
        new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
    else:
        new_width, new_height = int(default_max_size * old_width / old_height), default_max_size
    img = dlib.resize_image(img, rows=new_height, cols=new_width)

    # Detect faces
    dets = cnn_face_detector(img, 1)
    if len(dets) == 0:
        print("No face found in '{}'".format(image_path))
        return None

    # Detect face landmarks and align
    faces = dlib.full_object_detections()
    for detection in dets:
        rect = detection.rect
        faces.append(sp(img, rect))
    images = dlib.get_face_chips(img, faces, size=300, padding=0.25)

    # Assume we are processing only the first face (in case multiple faces are detected)

    def multi_img(aligned_face, ijk):
        # Convert to tensor for prediction
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        aligned_face_tensor = trans(aligned_face).unsqueeze(0)

        # Load the models and predict
        model_fair_7, device = load_models()
        aligned_face_tensor = aligned_face_tensor.to(device)

        # Predict with 7-class model
        outputs = model_fair_7(aligned_face_tensor).cpu().detach().numpy().squeeze()
        race_outputs, gender_outputs, age_outputs = outputs[:7], outputs[7:9], outputs[9:18]

        probabilities = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        sorted_indices = np.argsort(probabilities)[::-1]
        race_pred = np.argmax(np.exp(race_outputs) / np.sum(np.exp(race_outputs)))
        gender_pred = np.argmax(np.exp(gender_outputs) / np.sum(np.exp(gender_outputs)))
        age_pred = np.argmax(np.exp(age_outputs) / np.sum(np.exp(age_outputs)))

        # Map predictions to human-readable labels
        race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        gender_labels = ['male', 'female']
        age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

        for index in sorted_indices:
            if race_labels[index] == 'White':
                race_p = 'White'
                break
            elif race_labels[index] == 'Black':
                race_p = 'Black'
                break
            elif race_labels[index] == 'Latino_Hispanic':
                race_p = 'Hispanic'
                break
            elif race_labels[index] == 'East Asian' or race_labels[index] == 'Southeast Asian':
                race_p = 'Asian'
                break
            else:
                pass

        # if race_labels[race_pred] == 'White':
        #     race_p = 'White'
        # elif race_labels[race_pred] == 'Black':
        #     race_p = 'Black'
        # elif race_labels[race_pred] == 'Latino_Hispanic':
        #     race_p = 'Hispanic'
        # else:
        #     race_p = 'Asian'

        result = {
            "image": f'{image_path}[{ijk}]',
            "predicted_race": race_p,
            "predicted_gender": gender_labels[gender_pred],
            "predicted_age": age_labels[age_pred]
        }

        return result

    results = []
    for ijk, img_single in enumerate(images):
        results.append(multi_img(img_single, ijk))

    return results


if __name__ == '__main__':
    # Example usage
    image_path = '../tmp/test_muti_face.jpg'
    results = detect_and_predict_single_image(image_path)
    if results:
        print(len(results))
        for result in results:
            print(result)
