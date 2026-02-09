import numpy as np
from PIL import Image
import torch
import os
from pathlib import Path
from unet_baseline import UNet

def predict_mask(model, image_path, device, threshold=0.5, visualize=False):
    """
    Predict segmentation mask for a single image.
    
    Args:
        model: Trained UNet model
        image_path: Path to input CT image
        device: torch device
        threshold: Binarization threshold (default 0.5)
        visualize: If True, returns visualization data
        
    Returns:
        pred_mask: Binary mask (H, W) as numpy array
        If visualize=True, also returns (original_img, prob_map)
    """
    model.eval()
    
    # Load and preprocess image
    image = np.array(Image.open(image_path).convert('L').resize((384, 384), Image.BILINEAR)).astype(np.float32) / 255.0
    original_img = image.copy()
    
    # Add channel dimension if grayscale
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]
    else:
        image = np.transpose(image, (2, 0, 1))
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        prob_map = model(image_tensor).squeeze().cpu().numpy()
    
    # Binarize
    pred_mask = (prob_map > threshold).astype(np.float32)
    
    if visualize:
        return pred_mask, original_img, prob_map
    return pred_mask


def predict_and_visualize(model, image_path, mask_path, device, save_path=None):
    """
    Predict mask and create side-by-side visualization.
    
    Args:
        model: Trained UNet model
        image_path: Path to input CT image
        mask_path: Path to ground truth mask (optional, can be None)
        device: torch device
        save_path: If provided, saves visualization to this path
        
    Returns:
        Dictionary with metrics and visualization
    """
    import matplotlib.pyplot as plt
    
    # Get prediction
    pred_mask, original_img, prob_map = predict_mask(
        model, image_path, device, visualize=True
    )
    
    # Load ground truth if available
    metrics = {}
    if mask_path and os.path.exists(mask_path):
        gt_mask = np.array(Image.open(mask_path).resize((384, 384))).astype(np.float32) / 255.0
        
        # Calculate metrics
        intersection = (pred_mask * gt_mask).sum()
        union = (pred_mask + gt_mask).clip(0, 1).sum()
        pred_sum = pred_mask.sum()
        gt_sum = gt_mask.sum()
        
        metrics['iou'] = intersection / (union + 1e-6)
        metrics['dice'] = (2 * intersection) / (pred_sum + gt_sum + 1e-6)
        metrics['accuracy'] = ((pred_mask == gt_mask).sum() / pred_mask.size)
        metrics['coverage'] = intersection / (gt_sum + 1e-6)
    else:
        gt_mask = None
    
    # Create visualization
    # n_cols = 4 if gt_mask is not None else 3
    n_cols = 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original CT Image')
    axes[0].axis('off')
    
    # axes[1].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    # axes[1].set_title('Probability Map')
    # axes[1].axis('off')
    
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    if gt_mask is not None:
        axes[2].imshow(gt_mask, cmap='gray')
        # title = f'Ground Truth\nIoU: {metrics["iou"]:.3f} | Dice: {metrics["dice"]:.3f}'
        title = "Ground Truth"
        axes[2].set_title(title)
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.close()
    
    return {
        'pred_mask': pred_mask,
        'prob_map': prob_map,
        'metrics': metrics,
        'figure': fig
    }


def batch_predict(model, input_dir, output_dir, device, save_visualizations=True):
    """
    Predict masks for all images in a directory.
    
    Args:
        model: Trained UNet model
        input_dir: Directory containing *_ct.png images
        output_dir: Directory to save predictions
        device: torch device
        save_visualizations: If True, saves side-by-side comparisons
    """
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    if save_visualizations:
        os.makedirs(vis_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    image_files = sorted(list(input_path.glob("*_ct.png")))
    
    all_metrics = []
    
    for img_path in image_files:
        # Find corresponding mask
        mask_path = img_path.parent / img_path.name.replace("_ct.png", "_mask.png")
        
        # Predict
        pred_mask = predict_mask(model, str(img_path), device)
        
        # Save prediction
        pred_filename = img_path.name.replace("_ct.png", "_pred.png")
        pred_path = os.path.join(output_dir, pred_filename)
        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(pred_path)
        
        # Create visualization if requested
        if save_visualizations:
            vis_path = os.path.join(vis_dir, img_path.stem + "_comparison.png")
            result = predict_and_visualize(
                model, str(img_path), str(mask_path), device, save_path=vis_path
            )
            all_metrics.append({
                'filename': img_path.name,
                **result['metrics']
            })
    
    # Print summary statistics
    if all_metrics:
        avg_iou = np.mean([m['iou'] for m in all_metrics])
        avg_dice = np.mean([m['dice'] for m in all_metrics])
        avg_coverage = np.mean([m['coverage'] for m in all_metrics])
        
        print(f"\nBatch Prediction Summary:")
        print(f"Processed {len(all_metrics)} images")
        print(f"Avg IoU: {avg_iou:.4f}")
        print(f"Avg Dice: {avg_dice:.4f}")
        print(f"Avg Coverage: {avg_coverage:.4f}")
    
    print(f"\nPredictions saved to {output_dir}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Single image prediction
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load("results/drive+stare/best_unet.pth"))
    model.to(device)
    
    # # Just get the mask
    # mask = predict_mask(model, 
    #                     "results/drive+stare/best_unet.pth", 
    #                     device)
    
    # Get mask with visualization
    result = predict_and_visualize(
        model, 
        "../../../data/DRIVE/val/images/22_training.tif",
        "../../../data/DRIVE/val/segm/22_manual1.gif",
        device,
        save_path="results/drive+stare/prediction.png"
    )
    print(f"IoU: {result['metrics']['iou']:.3f}")
    
    # # Batch prediction on validation set
    # batch_predict(
    #     model, 
    #     "data/ct_like/2d/continuous/val",
    #     "classical_segmentation/predictions",
    #     device
    # )