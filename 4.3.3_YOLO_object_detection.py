import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():

    # 1. CONFIGURATION
    train_model = True
    validate_model = True

    # Path to dataset YAML file
    data_yaml_path = "C:/Users/Peter Zeng/Desktop/yolodata/dataset/data.yaml"

    # Where to save training runs and the final model
    model_output_dir = "C:/Users/Peter Zeng/Desktop/morevisualizations/yolodata/outcome/model"

    # Pretrained checkpoint for transfer learning
    pretrained_weights = "yolov8s.pt"

    # Final model name to save
    model_name = "box_snapfit_model"

    # Inference raw images folder
    raw_images_dir = "C:/Users/Peter Zeng/Desktop/yolodata/rawimages"

    # Where to save detection results
    inference_results_dir = "C:/Users/Peter Zeng/Desktop/morevisualizations/yolodata/outcome/predictions"

    # Where to save evaluation/metrics
    evaluation_dir = "C:/Users/Peter Zeng/Desktop/morevisualizations/yolodata/outcome/evaluation"

    # Directory for additional visualizations for the dissertation
    viz_dir = "C:/Users/Peter Zeng/Desktop/morevisualizations/yolodata/morevisualizations/"
    
    # Create output folders
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(inference_results_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # 2. TRAIN 
    if train_model:
        print("Starting training with YOLOv8 ...")
        # Initialize a YOLO object with a pretrained checkpoint
        model = YOLO(pretrained_weights)

        # Train on dataset and store the results
        train_results = model.train(
            data=data_yaml_path,          # path to data.yaml
            imgsz=640,                    # image size
            epochs=3,                    # number of training epochs
            batch=8,                      # adjust based on GPU/CPU memory
            name=model_name,              # name of this training run
            project=model_output_dir,     # folder where training results go
            exist_ok=True,                # overwrite if folder exists
            amp=False,                    # disable Automatic Mixed Precision
            lr0=1e-3                      # initial learning rate
        )
        print("Training completed.")

        # Visualization for training metrics:
        try:
            metrics = train_results[0].metrics  # hypothetical structure
            train_loss = metrics.get("train_loss", None)
            val_loss = metrics.get("val_loss", None)
            if train_loss is not None and val_loss is not None:
                epochs = range(1, len(train_loss) + 1)
                plt.figure()
                plt.plot(epochs, train_loss, label="Train Loss")
                plt.plot(epochs, val_loss, label="Validation Loss")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.title("Training and Validation Loss Curve")
                plt.legend()
                plt.savefig(os.path.join(viz_dir, "training_loss_curve.png"))
                plt.close()
                print("Training loss curve saved to:", os.path.join(viz_dir, "training_loss_curve.png"))
        except Exception as e:
            print("Could not generate training loss curve visualization:", e)
    else:
        print("Skipping training step...")

    # 3. EVALUATE 
    if validate_model:
        print("Starting validation/evaluation ...")

        best_weights_path = os.path.join(model_output_dir, model_name, "weights", "best.pt")

        # Initialize model from best weights
        model = YOLO(best_weights_path)

        # Validate on the validation set specified in data.yaml
        metrics = model.val()

        # Save the evaluation results (metrics) in a text file
        eval_file_path = os.path.join(evaluation_dir, "evaluation_metrics.txt")
        with open(eval_file_path, "w") as f:
            f.write(str(metrics))
        print(f"Evaluation metrics saved to: {eval_file_path}")

        # Visualization for evaluation metrics:
        try:
            eval_keys = ['precision', 'recall', 'mAP50', 'mAP50-95']
            eval_values = [metrics.get(k, 0) for k in eval_keys]
            plt.figure()
            plt.bar(eval_keys, eval_values, color=['blue', 'green', 'orange', 'red'])
            plt.xlabel("Metrics")
            plt.ylabel("Values")
            plt.title("Evaluation Metrics")
            plt.savefig(os.path.join(viz_dir, "evaluation_metrics_bar_chart.png"))
            plt.close()
            print("Evaluation metrics bar chart saved to:", os.path.join(viz_dir, "evaluation_metrics_bar_chart.png"))
        except Exception as e:
            print("Could not generate evaluation metrics visualization:", e)
    else:
        print("Skipping validation step...")

    # 4. INFERENCE (ON RAW IMAGES)
    print("Starting inference on raw images ...")

    # Decide which weights to use for inference:
    if train_model or validate_model:
        weights_path = os.path.join(model_output_dir, model_name, "weights", "best.pt")
    else:
        weights_path = pretrained_weights

    inference_model = YOLO(weights_path)

    # Find all images in the raw images directory
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    raw_image_paths = []
    for ext in image_extensions:
        raw_image_paths.extend(glob.glob(os.path.join(raw_images_dir, ext)))

    if not raw_image_paths:
        print(f"No images found in {raw_images_dir}. Please check the directory.")
        return

    print(f"Found {len(raw_image_paths)} images in {raw_images_dir}.")

    # For additional visualizations, collect detection confidences and counts
    all_confidences = []
    detection_counts = []
    for img_path in raw_image_paths:
        # Perform detection
        results = inference_model.predict(source=img_path, conf=0.25)
        # Collect detection confidences and counts
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                try:
                    # Convert tensor to list of confidences
                    confs = r.boxes.conf.cpu().numpy().tolist()
                except Exception as e:
                    confs = []
                all_confidences.extend(confs)
                detection_counts.append(len(confs))
            else:
                detection_counts.append(0)
        # Also save the annotated image
        inference_model.predict(
            source=img_path,
            conf=0.25,
            save=True,
            project=inference_results_dir,
            name="raw_inference",
            exist_ok=True
        )

    print(f"Inference completed. Results are saved to: {inference_results_dir}/raw_inference")

    # Visualization for Histogram of detection confidences
    try:
        if all_confidences:
            plt.figure()
            plt.hist(all_confidences, bins=20, edgecolor='black')
            plt.xlabel("Confidence Scores")
            plt.ylabel("Frequency")
            plt.title("Histogram of Detection Confidence Scores")
            plt.savefig(os.path.join(viz_dir, "detection_confidence_histogram.png"))
            plt.close()
            print("Detection confidence histogram saved to:", os.path.join(viz_dir, "detection_confidence_histogram.png"))
        if detection_counts:
            plt.figure()
            bins = range(0, max(detection_counts) + 2)
            plt.hist(detection_counts, bins=bins, align='left', edgecolor='black')
            plt.xlabel("Number of Detections per Image")
            plt.ylabel("Frequency")
            plt.title("Histogram of Detection Counts per Image")
            plt.savefig(os.path.join(viz_dir, "detection_counts_histogram.png"))
            plt.close()
            print("Detection counts histogram saved to:", os.path.join(viz_dir, "detection_counts_histogram.png"))
    except Exception as e:
        print("Could not generate inference visualizations:", e)

    # Additional visualization Create a montage of sample annotated inference images
    annotated_images_dir = os.path.join(inference_results_dir, "raw_inference")
    annotated_image_paths = glob.glob(os.path.join(annotated_images_dir, "*.jpg"))
    if annotated_image_paths:
        sample_images = annotated_image_paths[:9]  # take first 9 images for montage
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for ax, img_file in zip(axes.flatten(), sample_images):
            img = cv2.imread(img_file)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(os.path.basename(img_file))
                ax.axis('off')
        plt.suptitle("Sample Annotated Inference Results")
        plt.tight_lamet()
        plt.savefig(os.path.join(viz_dir, "inference_montage.png"))
        plt.close()
        print("Inference montage saved to:", os.path.join(viz_dir, "inference_montage.png"))

    print("All done!")

if __name__ == "__main__":
main()
