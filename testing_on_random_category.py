
def generate_random_doodle(size=128, max_strokes=5):
    """Generate a random doodle-like image"""
    img = np.ones((size, size), dtype=np.float32)
    # Random number of strokes (1 to max_strokes)
    num_strokes = np.random.randint(1, max_strokes+1)
    
    for _ in range(num_strokes):
        # Randomly choose between line, circle, or ellipse
        shape_type = np.random.choice(['line', 'circle', 'ellipse'])
        
        if shape_type == 'line':
            # Generate random line
            r0, c0 = np.random.randint(0, size, 2)
            r1, c1 = np.random.randint(0, size, 2)
            rr, cc = line(r0, c0, r1, c1)
            
        elif shape_type == 'circle':
            # Generate random circle using disk
            radius = np.random.randint(5, size//4)
            center = np.random.randint(radius, size - radius, 2)
            rr, cc = disk((center[0], center[1]), radius, shape=(size, size))
            
        else:  # ellipse
            # Generate random ellipse
            r_radius = np.random.randint(5, size//4)
            c_radius = np.random.randint(5, size//4)
            center = np.random.randint(max(r_radius, c_radius), size - max(r_radius, c_radius), 2)
            rr, cc = ellipse(center[0], center[1], r_radius, c_radius, shape=(size, size))
        
        # Ensure coordinates are within bounds
        rr = np.clip(rr, 0, size-1)
        cc = np.clip(cc, 0, size-1)
        
        # Draw the shape: set pixels to 0 (black) on the white background
        img[rr, cc] = 0
    
    return img

def create_partial_doodle(doodle, max_remove=2):
    """Remove random strokes to create partial input"""
    # Find all connected components (strokes)
    from skimage.measure import label
    labeled = label(1 - doodle, connectivity=2)
    
    # Get unique stroke IDs (excluding background)
    stroke_ids = np.unique(labeled)
    stroke_ids = stroke_ids[stroke_ids != 0]
    
    if len(stroke_ids) <= 1:
        return doodle  # Can't remove if only 1 stroke
    
    # Randomly remove strokes
    num_remove = np.random.randint(1, min(max_remove, len(stroke_ids)) + 1)
    remove_ids = np.random.choice(stroke_ids, num_remove, replace=False)
    
    # Create mask for strokes to keep
    mask = ~np.isin(labeled, remove_ids)
    partial = np.ones_like(doodle)
    partial[mask] = doodle[mask]
    
    return partial

def test_random_input(generator, num_tests=3):
    """Generate and test random doodles"""
    plt.figure(figsize=(15, 5*num_tests))
    
    for i in range(num_tests):
        # 1. Generate random complete doodle
        complete = generate_random_doodle()
        
        # 2. Create partial version
        partial = create_partial_doodle(complete)
        
        # 3. Prepare for model (normalize to [-1,1])
        model_input = (partial * 2) - 1
        model_input = model_input[np.newaxis, ..., np.newaxis]  # Add batch and channel dims
        
        # 4. Generate prediction
        generated = generator.predict(model_input)[0, ..., 0]  # Remove batch/channel dims
        generated = (generated + 1) / 2  # Convert back to [0,1]
        
        # Plot results
        plt.subplot(num_tests, 4, i*4+1)
        plt.imshow(partial, cmap='gray')
        plt.title(f"Test {i+1}\nPartial Input")
        plt.axis('off')
        
        plt.subplot(num_tests, 4, i*4+2)
        plt.imshow(generated, cmap='gray')
        plt.title("Model Output")
        plt.axis('off')
        
        plt.subplot(num_tests, 4, i*4+3)
        plt.imshow(complete, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
        
        plt.subplot(num_tests, 4, i*4+4)
        plt.imshow(np.abs(generated - complete), cmap='hot')
        plt.title("Difference")
        plt.colorbar()
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example Usage:
if __name__ == "__main__":
    # Load your trained model
    generator = tf.keras.models.load_model(
        "doodle_completer_final.h5",
        compile=False,
        custom_objects={'SelfAttention': SelfAttention}
    )
    
    # Test with random inputs
    test_random_input(generator, num_tests=3)