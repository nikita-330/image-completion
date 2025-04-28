# TESTING ON TRAINING CATEGORIES

def test_on_training_categories(model_path, num_samples_per_category=3):
    
    # 1. LOAD MODEL
    custom_objects = {'SelfAttention': SelfAttention}
    generator = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects=custom_objects
    )
    
    # 1. PREPARE CATEGORY-SPECIFIC TEST DATA
    categories = [
        "The Eiffel Tower",
        "The Mona Lisa",
        "car", 
        "apple",
        "baseball",
        "alarm clock"
    ]
    
    loader = QuickDrawLoader()  # Reuse your data loader
    all_results = []
    
    for category in categories:
        print(f"\n🔍 Testing on: {category}")
        
        # Load specific category data
        df = pd.read_csv(os.path.join(config.DATA_PATH, f"{category}.csv"), 
                        nrows=num_samples_per_category*2)  # Get extra for variety
        
        # Process samples
        samples = []
        for _, row in df.iterrows():
            strokes = eval(row['drawing'])
            partial = loader.strokes_to_image(loader._remove_random_strokes(strokes))
            complete = loader.strokes_to_image(strokes)
            samples.append((partial, complete))
        
        # Select random samples
        selected_samples = np.random.choice(len(samples), num_samples_per_category, replace=False)
        
        # Generate and display results
        category_results = []
        for idx in selected_samples:
            partial_img = (samples[idx][0] * 2) - 1  # Normalize to [-1,1]
            partial_img = np.expand_dims(partial_img, axis=(0, -1))  # Add batch and channel dims
            
            # Generate completion
            generated = generator.predict(partial_img)[0]
            
            # Store results
            result = {
                'category': category,
                'partial': samples[idx][0],
                'ground_truth': samples[idx][1],
                'generated': generated.squeeze(),
                'processed': post_process(generated)
            }
            category_results.append(result)
            
            # Display
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            titles = ['Partial Input', 'Ground Truth', 'Raw Output', 'Processed']
            images = [result['partial'], result['ground_truth'], 
                     result['generated'], result['processed']]
            
            for ax, img, title in zip(axes, images, titles):
                ax.imshow(img, cmap='gray')
                ax.set_title(f"{category}\n{title}")
                ax.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        all_results.extend(category_results)
    
    return all_results

# RUN THE TEST
test_results = test_on_training_categories("doodle_completer_final.h5", num_samples_per_category=2)