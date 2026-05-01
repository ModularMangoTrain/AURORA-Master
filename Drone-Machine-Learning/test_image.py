import torch
from torchvision import transforms
from PIL import Image
import sys

# Load model
print("Loading model...")
spatial = torch.load("spatial_person_detector_quantized.pth", map_location=torch.device('cpu'), weights_only=False)
device = torch.device("cpu")
spatial.to(device)
spatial.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def test_image(image_path):
    print(f"\nTesting: {image_path}")
    
    # Load image
    image = Image.open(image_path)
    
    # Check if it's BGR (from OpenCV) or RGB
    # OpenCV saves as BGR, so convert if needed
    if image_path.startswith('frame_'):
        # Saved from OpenCV, convert BGR to RGB
        import numpy as np
        img_array = np.array(image)
        img_array = img_array[:, :, ::-1]  # BGR to RGB
        image = Image.fromarray(img_array)
    
    image = image.convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.inference_mode():
        output = spatial(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    classes = ["no_person", "person"]
    prediction = classes[predicted_idx.item()]
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence.item():.2%}")
    print(f"Raw scores - no_person: {probabilities[0].item():.4f}, person: {probabilities[1].item():.4f}")
    
    return prediction, confidence.item()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <image_path>")
        print("Example: python test_image.py test.jpg")
        exit(1)
    
    image_path = sys.argv[1]
    test_image(image_path)
