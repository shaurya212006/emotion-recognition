import argparse, cv2, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils.predictor import process_image

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to image file")
parser.add_argument("--show",  action="store_true", help="Show result window")
parser.add_argument("--save",  default="", help="Save annotated image to path")
args = parser.parse_args()

image = cv2.imread(args.image)
if image is None:
    print(f"Cannot read: {args.image}"); sys.exit(1)

annotated, results = process_image(image)

if not results:
    print("No face detected.")
else:
    for i, r in enumerate(results, 1):
        print(f"Face {i}: {r['label']} ({r['confidence']*100:.1f}%)")

if args.save:
    cv2.imwrite(args.save, annotated)
    print(f"Saved to: {args.save}")

if args.show:
    cv2.imshow("EmoScan", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()