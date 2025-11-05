"""
Quick Dataset Analysis Script
Test your health insurance dataset integration
"""
import os
import sys

print("🏥 Health Insurance Dataset Integration Test")
print("=" * 50)

# Check if dataset file exists
dataset_path = "../data/health_insurance_dataset.csv"
if os.path.exists(dataset_path):
    print("✅ Dataset file found!")
    print(f"📍 Location: {os.path.abspath(dataset_path)}")
    
    try:
        # Try to read with basic Python (no pandas needed)
        with open(dataset_path, 'r') as file:
            lines = file.readlines()
            print(f"📊 Dataset has {len(lines)} lines")
            print(f"📋 Header: {lines[0].strip()}")
            
            if len(lines) > 1:
                print(f"📝 Sample data: {lines[1].strip()}")
                
        print("\n🎯 Your dataset is ready for integration!")
        print("\nNext steps:")
        print("1. Open demo.html in your browser")
        print("2. Test the fraud detection form")
        print("3. The backend API should connect to your data")
        
    except Exception as e:
        print(f"⚠️ Error reading dataset: {e}")
        
else:
    print("❌ Dataset file not found!")
    print(f"📍 Expected location: {os.path.abspath(dataset_path)}")
    print("\n📋 To fix this:")
    print("1. Move your health_insurance_dataset.csv to the data/ folder")
    print("2. Ensure the filename matches exactly")

print("\n" + "=" * 50)
print("🚀 Ready to test your fraud detection system!")
print("Open demo.html to access the frontend interface")