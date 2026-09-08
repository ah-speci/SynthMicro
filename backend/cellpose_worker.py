import sys
import numpy as np
import cv2

from cellpose import models


def main():
    if len(sys.argv) != 3:
        print("Usage: python cellpose_worker.py INPUT_IMAGE OUTPUT_MASK")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print("Cellpose worker starting...")
    print("Input:", input_path)

    image = cv2.imread(input_path)

    if image is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    print("Image shape:", image.shape)

    model = models.CellposeModel(
        gpu=False,
        model_type="cpsam",
    )

    print("Cellpose model loaded.")

    masks, flows, styles = model.eval(
        image,
        channels=[0, 0],
    )

    masks = np.asarray(masks)

    print("Masks shape:", masks.shape)
    print("Objects:", int(masks.max()))

    np.save(output_path, masks)

    print("Mask saved:", output_path)


if __name__ == "__main__":
    main()
