def calculate_receptive_field(layers):
    r = 1  # Initial receptive field size
    s = 1  # Initial stride
    p = 0  # Initial padding

    for layer in layers:
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            k = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
            stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
            padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)

            r = r + (k[0] - 1) * s
            s = s * stride[0]
            p = p + padding[0]

    return {"receptive_field": r, "stride": s, "padding": p}

feature_maps = {}

def hook_fn(module, input, output):
    feature_maps[module] = output

for layer in model.features:
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
        layer.register_forward_hook(hook_fn)

def map_feature_to_input(output_coord, receptive_field, stride, padding):
    h_out, w_out = output_coord

    # Center of the receptive field
    h_in = h_out * stride - padding + (receptive_field - 1) // 2
    w_in = w_out * stride - padding + (receptive_field - 1) // 2

    # Boundaries of the receptive field
    h_start = h_in - (receptive_field - 1) // 2
    h_end = h_in + (receptive_field - 1) // 2
    w_start = w_in - (receptive_field - 1) // 2
    w_end = w_in + (receptive_field - 1) // 2

    return {"center": (h_in, w_in), "boundaries": {"start": (h_start, w_start), "end": (h_end, w_end)}}

import cv2
import numpy as np

def draw_bounding_boxes(image, feature_map, receptive_field, stride, padding):
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert tensor to HWC format
    image = (image * 255).astype(np.uint8)

    for h_out in range(feature_map.size(2)):
        for w_out in range(feature_map.size(3)):
            mapping = map_feature_to_input((h_out, w_out), receptive_field, stride, padding)
            start = mapping["boundaries"]["start"]
            end = mapping["boundaries"]["end"]

            # Draw rectangle on the image
            cv2.rectangle(image, (int(start[1]), int(start[0])), (int(end[1]), int(end[0])), (255, 0, 0), 1)

    # Save or display the image
    cv2.imshow("Mapped Features", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


input_image, _ = test_dataset[0]  # Load one image
input_image = input_image.unsqueeze(0).to(device)  # Add batch dimension

output = model(input_image)  # Forward pass

# Get the last feature map from hooks
last_feature_map = list(feature_maps.values())[-1]

# Calculate receptive field
layer_details = calculate_receptive_field(model.features)
r, s, p = layer_details["receptive_field"], layer_details["stride"], layer_details["padding"]

# Draw bounding boxes
draw_bounding_boxes(input_image[0], last_feature_map, r, s, p)

